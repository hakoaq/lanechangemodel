# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import subprocess
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traci
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional, Any
import socket
import traceback
import collections
import math
import copy # For deep copying configs/normalizers

# Solve Chinese garbled characters in matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

# --- Import necessary components from training scripts ---
try:
    # Assuming dqn.py and ppo.py are in the same directory or Python path
    from dqn import Config as DQN_Config
    from dqn import QNetwork, NoisyLinear, RunningMeanStd as DQN_RunningMeanStd, SumoEnv as DQN_SumoEnv # Use one SumoEnv, but need configs/nets
    # Note: DQN uses its own RewardNormalizer, not needed directly for evaluation logic but part of env
    # Note: DQN's agent logic is integrated into the evaluation loop

    from ppo import Config as PPO_Config
    from ppo import PPO, BehaviorCloningNet # BC Net not strictly needed but part of ppo.py
    from ppo import RunningMeanStd as PPO_RunningMeanStd, SumoEnv as PPO_SumoEnv # Need PPO's Normalizer
    # Note: PPO uses its own RewardNormalizer, not needed directly for evaluation logic
    # Note: PPO's agent logic (get_action) is adapted below

except ImportError as e:
    print(f"Error importing from dqn.py or ppo.py: {e}")
    print("Please ensure dqn.py and ppo.py are in the same directory or accessible in the Python path.")
    sys.exit(1)

#####################################
# Evaluation Configuration          #
#####################################
class EvalConfig:
    # --- Model Paths ---
    # IMPORTANT: Update these paths to your saved model files
    DQN_MODEL_PATH = "dqn.pth" # CHANGE ME
    PPO_MODEL_PATH = "ppo.pth"              # CHANGE ME

    # --- SUMO Configuration ---
    EVAL_SUMO_BINARY = "sumo"  # Use GUI for visualization during evaluation
    EVAL_SUMO_CONFIG = "new.sumocfg" # Use the new configuration file
    EVAL_STEP_LENGTH = 0.2         # Should match training step length
    EVAL_PORT_RANGE = (8910, 8920) # Use a different port range than training

    # --- Evaluation Parameters ---
    EVAL_EPISODES = 20             # Number of episodes to run for each model
    EVAL_MAX_STEPS = 1500          # Max steps per evaluation episode (e.g., 300 seconds)
    EVAL_SEED = 42                 # Seed for reproducibility if needed (applies to SumoEnv start)
    NUM_LANES = 4                  # Number of lanes in new.net.xml (0, 1, 2, 3)

    # --- Forced Lane Change Attempt Logic ---
    FORCE_CHANGE_INTERVAL_STEPS = 75 # Attempt a forced change every X steps
    FORCE_CHANGE_MONITOR_STEPS = 15  # How many steps to wait/monitor for the change to complete
    FORCE_CHANGE_SUCCESS_DIST = 5.0  # Min distance moved laterally for success check

    # --- Normalization ---
    # Load normalization settings from original configs, but manage state here
    # Assumes both models used normalization if their respective Config said so.

#####################################
# Helper Functions (Copied/Adapted) #
#####################################
def get_available_port(start_port, end_port):
    """Find an available port within the specified range"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError(f"No available port found in the range [{start_port}, {end_port}].")

def kill_sumo_processes():
    """Kill any lingering SUMO processes"""
    killed = False
    try:
        if os.name == 'nt': # Windows
            result1 = os.system("taskkill /f /im sumo.exe >nul 2>&1")
            result2 = os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
            killed = (result1 == 0 or result2 == 0)
        else: # Linux/macOS
            result1 = os.system("pkill -f sumo > /dev/null 2>&1")
            result2 = os.system("pkill -f sumo-gui > /dev/null 2>&1")
            killed = (result1 == 0 or result2 == 0)
    except Exception as e: print(f"Warning: Error terminating SUMO processes: {e}")
    time.sleep(0.1)

# Use the RunningMeanStd from DQN script (identical to PPO's version)
RunningMeanStd = DQN_RunningMeanStd

#####################################
# Evaluation Environment Wrapper    #
#####################################
# Using a slightly modified SumoEnv, inheriting most from DQN version for consistency
# Removed reward calculation complexity, focuses on state and execution
class EvaluationEnv:
    def __init__(self, eval_config: EvalConfig, sumo_seed: int):
        self.config = eval_config # Use evaluation config
        self.sumo_binary = self.config.EVAL_SUMO_BINARY
        self.config_path = self.config.EVAL_SUMO_CONFIG
        self.step_length = self.config.EVAL_STEP_LENGTH
        # Assume ego vehicle ID is the same as in training files
        self.ego_vehicle_id = DQN_Config.ego_vehicle_id
        self.ego_type_id = DQN_Config.ego_type_id
        self.port_range = self.config.EVAL_PORT_RANGE
        self.num_lanes = self.config.NUM_LANES
        self.sumo_seed = sumo_seed

        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_raw_state = np.zeros(DQN_Config.state_dim) # Assuming state dim is same
        self.current_step = 0
        self.collision_occurred = False
        self.ego_start_pos = None
        self.ego_route_id = "route_E0" # Assumes route ID from new.rou.xml

    def _start_sumo(self):
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.port_range[0], self.port_range[1])
        except IOError as e:
             print(f"ERROR: Failed to find available port: {e}")
             sys.exit(1)

        sumo_cmd = [
            self.sumo_binary, "-c", self.config_path,
            "--remote-port", str(self.traci_port),
            "--step-length", str(self.step_length),
            "--collision.check-junctions", "true",
            "--collision.action", "warn",
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--seed", str(self.sumo_seed) # Use provided seed
        ]
        if self.sumo_binary == "sumo-gui":
            sumo_cmd.extend(["--quit-on-end", "false"]) # Keep GUI open after sim ends

        try:
             stdout_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
             print(f"Starting SUMO on port {self.traci_port} with seed {self.sumo_seed}...")
        except FileNotFoundError:
             print(f"ERROR: SUMO executable '{self.sumo_binary}' not found.")
             sys.exit(1)
        except Exception as e:
             print(f"ERROR: Failed to start SUMO process: {e}")
             sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5)
                traci.init(self.traci_port)
                print(f"SUMO TraCI connected (Port: {self.traci_port}).")
                return
            except traci.exceptions.TraCIException:
                if attempt == connection_attempts - 1:
                    print("Max TraCI connection attempts reached.")
                    self._close()
                    raise ConnectionError(f"Could not connect to SUMO (Port: {self.traci_port}).")
            except Exception as e:
                print(f"Unexpected error connecting TraCI: {e}")
                self._close()
                raise ConnectionError(f"Unknown error connecting to SUMO (Port: {self.traci_port}).")

    def _add_ego_vehicle(self):
        """Add the ego vehicle to the simulation using settings from training config"""
        # Check route (assuming 'route_E0' from new.rou.xml)
        if self.ego_route_id not in traci.route.getIDList():
            edge_list = list(traci.edge.getIDList())
            first_edge = edge_list[0] if edge_list else None
            if first_edge:
                print(f"Warning: Route '{self.ego_route_id}' not found. Creating route from first edge '{first_edge}'.")
                try:
                    traci.route.add(self.ego_route_id, [first_edge])
                except traci.exceptions.TraCIException as e:
                    raise RuntimeError(f"Failed to add route '{self.ego_route_id}' using edge '{first_edge}': {e}")
            else:
                raise RuntimeError(f"Route '{self.ego_route_id}' not found and no edges available to create it.")


        # Check type
        if self.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.ego_type_id) # Copy from default 'car' type in new.rou.xml
                traci.vehicletype.setParameter(self.ego_type_id, "color", "1,0,0") # Red
                # Apply key parameters from training config if needed (optional for eval)
                # traci.vehicletype.setParameter(self.ego_type_id, "lcStrategic", "1.0")
            except traci.exceptions.TraCIException as e:
                print(f"Warning: Failed to set parameters for Ego type '{self.ego_type_id}': {e}")

        # Remove residual ego
        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"Warning: Failed to remove residual Ego: {e}")

        # Add ego
        try:
            start_lane = random.choice(range(self.num_lanes)) # Random start lane
            traci.vehicle.add(vehID=self.ego_vehicle_id, routeID=self.ego_route_id,
                              typeID=self.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max")

            # Wait for ego to appear
            wait_steps = int(2.0 / self.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.ego_vehicle_id in traci.vehicle.getIDList():
                    ego_appeared = True
                    self.ego_start_pos = traci.vehicle.getPosition(self.ego_vehicle_id)
                    break
            if not ego_appeared:
                raise RuntimeError(f"Ego vehicle '{self.ego_vehicle_id}' did not appear within {wait_steps} steps.")

        except traci.exceptions.TraCIException as e:
            print(f"ERROR: Failed to add Ego vehicle '{self.ego_vehicle_id}': {e}")
            raise RuntimeError("Failed adding Ego vehicle.")

    def reset(self) -> np.ndarray:
        """Reset environment for a new evaluation episode"""
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.current_step = 0
        self.collision_occurred = False
        self.last_raw_state = np.zeros(DQN_Config.state_dim)

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                raw_state = self._get_raw_state()
                self.last_raw_state = raw_state.copy()
            except traci.exceptions.TraCIException:
                 print("Warning: TraCI exception during initial state fetch in reset.")
        else:
             print("Warning: Ego vehicle not found immediately after add/wait in reset.")

        return self.last_raw_state

    # Reusing _get_surrounding_vehicle_info logic from dqn.py (identical to ppo.py)
    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """Get distance and relative speed of nearest vehicles (copied from dqn.py)"""
        max_dist = DQN_Config.max_distance # Use distance from a config
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        ego_road = traci.vehicle.getRoadID(ego_id)
        if not ego_road: return infos # Check if on a road

        # Get number of lanes on current edge dynamically if needed, or use EvalConfig.NUM_LANES
        # num_lanes_on_edge = traci.edge.getLaneNumber(ego_road)

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                # Check if the vehicle is on the same edge first
                veh_road = traci.vehicle.getRoadID(veh_id)
                if veh_road != ego_road: continue

                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                # Simple longitudinal distance assumption for state
                dx = veh_pos[0] - ego_pos[0]
                # Lateral distance check might be useful too: dy = veh_pos[1] - ego_pos[1]
                distance = abs(dx) # Use longitudinal distance for state representation consistency

                if distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed # Positive if ego is faster

                # Determine relative position based on lanes and dx
                if veh_lane == ego_lane: # Same lane
                    if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                    # Ignore vehicles directly behind in same lane for this state representation
                elif veh_lane == ego_lane - 1: # Left lane
                    if dx > -5 and distance < infos['left_front'][0]: # Check vehicles slightly behind to front
                        infos['left_front'] = (distance, rel_speed)
                    elif dx <= -5 and distance < infos['left_back'][0]: # Check vehicles further behind
                        infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1: # Right lane
                     if dx > -5 and distance < infos['right_front'][0]:
                        infos['right_front'] = (distance, rel_speed)
                     elif dx <= -5 and distance < infos['right_back'][0]:
                        infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue # Skip vehicle if TraCI error occurs
        return infos

    # Reusing _get_raw_state logic from dqn.py
    def _get_raw_state(self) -> np.ndarray:
        """Get the current environment state (RAW values before normalization, copied from dqn.py)"""
        state = np.zeros(DQN_Config.state_dim, dtype=np.float32)
        ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            return self.last_raw_state # Return last known raw state if ego vanished

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road:
                return self.last_raw_state # Not on a road

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            num_lanes = self.num_lanes # Use known number of lanes

            # Ensure ego_lane is valid
            if not (0 <= ego_lane < num_lanes):
                 print(f"Warning: Invalid ego lane {ego_lane} detected. Clipping.")
                 ego_lane = np.clip(ego_lane, 0, num_lanes - 1)

            # Check lane change possibility (TraCI handles boundary checks internally)
            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1)
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1)

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # --- Raw State Features (Matches training state) ---
            state[0] = ego_speed
            state[1] = float(ego_lane)
            state[2] = min(surround_info['front'][0], DQN_Config.max_distance)
            state[3] = surround_info['front'][1]
            state[4] = min(surround_info['left_front'][0], DQN_Config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], DQN_Config.max_distance)
            state[7] = min(surround_info['right_front'][0], DQN_Config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], DQN_Config.max_distance)
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"Warning: NaN or Inf detected in raw state calculation. Using last valid raw state.")
                return self.last_raw_state

            self.last_raw_state = state.copy() # Store the latest valid raw state

        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" in str(e): pass # Expected near end
            else: print(f"Warning: TraCI error getting raw state for {ego_id}: {e}. Returning last known raw state.")
            return self.last_raw_state
        except Exception as e:
            print(f"Warning: Unknown error getting raw state for {ego_id}: {e}. Returning last known raw state.")
            traceback.print_exc()
            return self.last_raw_state

        return state

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """Execute action, return (next_raw_state, done)"""
        done = False
        ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
             self.collision_occurred = True # Assume collision if missing at step start
             return self.last_raw_state, True

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id)
            num_lanes = self.num_lanes

            # Ensure current_lane is valid before action execution
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)

            # 1. Execute Action
            if action == 1 and current_lane > 0: # Try Left
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
            elif action == 2 and current_lane < (num_lanes - 1): # Try Right
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
            # Action 0: Keep lane (do nothing explicit)

            # 2. Simulation Step
            traci.simulationStep()
            self.current_step += 1

            # 3. Check Status AFTER step
            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True
                done = True
                next_raw_state = self.last_raw_state # Return last known state
                return next_raw_state, done

            # Check SUMO collision list
            collisions = traci.simulation.getCollisions()
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True
                    done = True
                    break # No need to check further

            # Get next state (raw)
            next_raw_state = self._get_raw_state()

            # Check other termination conditions
            if traci.simulation.getTime() >= 3600: done = True # Sim time limit
            if self.current_step >= self.config.EVAL_MAX_STEPS: done = True # Eval step limit

        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" in str(e):
                self.collision_occurred = True # Assume collision if vehicle disappears during step
            else: print(f"ERROR: TraCI exception during step {self.current_step}: {e}")
            done = True
            next_raw_state = self.last_raw_state
        except Exception as e:
            print(f"ERROR: Unknown exception during step {self.current_step}: {e}")
            traceback.print_exc()
            done = True
            self.collision_occurred = True # Assume collision on unknown error
            next_raw_state = self.last_raw_state

        # Return RAW state and done flag
        return next_raw_state, done

    def get_vehicle_info(self):
        """Get current info like speed, lane, position"""
        ego_id = self.ego_vehicle_id
        if ego_id in traci.vehicle.getIDList():
            try:
                speed = traci.vehicle.getSpeed(ego_id)
                lane = traci.vehicle.getLaneIndex(ego_id)
                pos = traci.vehicle.getPosition(ego_id)
                dist_traveled = math.dist(pos, self.ego_start_pos) if self.ego_start_pos else 0.0
                return {"speed": speed, "lane": lane, "pos": pos, "dist": dist_traveled}
            except traci.exceptions.TraCIException:
                return {"speed": 0, "lane": -1, "pos": (0,0), "dist": 0}
        else:
            return {"speed": 0, "lane": -1, "pos": (0,0), "dist": 0}

    def _close(self):
        """Close SUMO instance"""
        if self.sumo_process:
            try:
                traci.close()
            except Exception: pass
            finally:
                try:
                    if self.sumo_process.poll() is None:
                        self.sumo_process.terminate()
                        self.sumo_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.sumo_process.kill()
                    self.sumo_process.wait(timeout=1)
                except Exception as e: print(f"Warning: Error during SUMO termination: {e}")
                self.sumo_process = None
                self.traci_port = None
                time.sleep(0.1)
        else:
            self.traci_port = None

#####################################
# Model Loading and Action Selection #
#####################################

def load_model(model_path: str, model_type: str, config: Any, device: torch.device) -> nn.Module:
    """Loads a trained model (DQN or PPO)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type == 'dqn':
        # Ensure config is an instance of DQN_Config
        if not isinstance(config, DQN_Config): # CORRECTED CHECK
             # Add a print statement to see what type it actually is, for debugging
             print(f"Debug: Expected DQN_Config, but got type: {type(config)}")
             raise TypeError("Provided config is not an instance of DQN_Config for DQN model")
        model = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(device)
    elif model_type == 'ppo':
        # Ensure config is an instance of PPO_Config
        if not isinstance(config, PPO_Config): # CORRECTED CHECK
             # Add a print statement for debugging
             print(f"Debug: Expected PPO_Config, but got type: {type(config)}")
             raise TypeError("Provided config is not an instance of PPO_Config for PPO model")
        model = PPO(config.state_dim, config.action_dim, config.hidden_size).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode
        print(f"Successfully loaded {model_type.upper()} model from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        raise

def normalize_state(state_raw: np.ndarray, normalizer: Optional[RunningMeanStd], clip_val: float) -> np.ndarray:
    """Normalizes state using the provided normalizer instance."""
    if normalizer:
        # Update normalizer stats during evaluation (suboptimal but necessary if not saved)
        # Use a copy to avoid modifying the original normalizer across models if shared
        temp_normalizer = copy.deepcopy(normalizer)
        temp_normalizer.update(state_raw.reshape(1, -1)) # Update with single observation

        norm_state = (state_raw - temp_normalizer.mean) / (temp_normalizer.std + 1e-8)
        norm_state = np.clip(norm_state, -clip_val, clip_val)
        return norm_state.astype(np.float32), temp_normalizer # Return updated normalizer
    else:
        return state_raw.astype(np.float32), normalizer # Return raw state if no normalizer

def get_dqn_action(model: QNetwork, state_norm: np.ndarray, current_lane_idx: int, config: DQN_Config, device: torch.device) -> int:
    """Get action from DQN model (C51/Noisy aware)."""
    model.eval() # Ensure eval mode

    # Reset noise if using Noisy Nets
    if config.use_noisy_nets:
        for module in model.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        action_probs = model(state_tensor) # Output: [1, action_dim, num_atoms]
        support = torch.linspace(config.v_min, config.v_max, config.num_atoms).to(device)
        expected_q_values = (action_probs * support).sum(dim=2) # [1, action_dim]

        # Action Masking
        q_values_masked = expected_q_values.clone()
        if current_lane_idx == 0:
            q_values_masked[0, 1] = -float('inf') # Cannot turn left from lane 0
        if current_lane_idx >= EvalConfig.NUM_LANES - 1:
            q_values_masked[0, 2] = -float('inf') # Cannot turn right from last lane

        action = q_values_masked.argmax().item()
    return action

def get_ppo_action(model: PPO, state_norm: np.ndarray, current_lane_idx: int, device: torch.device) -> int:
    """Get deterministic action from PPO actor."""
    model.eval() # Ensure eval mode
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        # Get action probabilities from the actor part
        action_probs = model.get_action_probs(state_tensor) # [1, action_dim]

        # Action Masking
        probs_masked = action_probs.clone()
        if current_lane_idx == 0:
            probs_masked[0, 1] = -float('inf') # Effectively mask by setting prob to ~0
        if current_lane_idx >= EvalConfig.NUM_LANES - 1:
            probs_masked[0, 2] = -float('inf')

        # Choose action with highest probability after masking
        action = probs_masked.argmax().item()
    return action


#####################################
# Evaluation Episode Runner         #
#####################################

EpisodeResult = namedtuple('EpisodeResult', [
    'steps', 'collided', 'avg_speed', 'total_dist',
    'forced_attempts', 'forced_agreed', 'forced_executed_safe', 'forced_executed_collision',
    'model_lane_changes'
])

def evaluate_episode(
    model: nn.Module,
    model_type: str, # 'dqn' or 'ppo'
    env: EvaluationEnv,
    config: Any, # DQN_Config or PPO_Config
    obs_normalizer: Optional[RunningMeanStd], # Initial normalizer state
    device: torch.device,
    eval_config: EvalConfig
) -> Tuple[EpisodeResult, Optional[RunningMeanStd]]:
    """Runs a single evaluation episode for a given model."""

    state_raw = env.reset()
    # Crucial: Create a deep copy of the normalizer for this episode run
    current_obs_normalizer = copy.deepcopy(obs_normalizer)

    done = False
    step_count = 0
    speeds = []
    model_lane_changes = 0 # Count changes initiated by model action != 0

    # Forced change tracking
    forced_attempts = 0
    forced_agreed = 0           # Model chose the desired action
    forced_executed_safe = 0    # Model agreed, change completed, no collision
    forced_executed_collision = 0 # Model agreed, change attempted, collision occurred
    monitoring_change = False
    monitor_steps_left = 0
    monitor_target_action = -1
    monitor_start_lane = -1
    monitor_start_pos = None

    while not done and step_count < eval_config.EVAL_MAX_STEPS:
        # 1. Normalize state (updates and returns the episode-specific normalizer)
        if model_type == 'dqn':
             state_norm, current_obs_normalizer = normalize_state(state_raw, current_obs_normalizer, config.obs_norm_clip)
        elif model_type == 'ppo':
             state_norm, current_obs_normalizer = normalize_state(state_raw, current_obs_normalizer, config.obs_norm_clip)
        else: # Should not happen
             state_norm = state_raw
             print("Warning: Unknown model type for normalization")


        # 2. Get current lane and decide on forced change attempt
        current_lane_idx = int(round(state_raw[1])) # Index 1 is lane index
        current_lane_idx = np.clip(current_lane_idx, 0, eval_config.NUM_LANES - 1)
        can_go_left = current_lane_idx > 0
        can_go_right = current_lane_idx < (eval_config.NUM_LANES - 1)
        target_action = -1 # -1 means no forced action

        # Check if monitoring a previous attempt
        if monitoring_change:
            monitor_steps_left -= 1
            current_pos = env.get_vehicle_info()['pos']
            current_lane_after_step = env.get_vehicle_info()['lane']

            # Check for successful physical change (significant lateral movement or lane index change)
            lateral_dist = abs(current_pos[1] - monitor_start_pos[1])
            lane_changed_physically = (current_lane_after_step != monitor_start_lane)

            # Check completion criteria
            if lane_changed_physically or lateral_dist > eval_config.FORCE_CHANGE_SUCCESS_DIST:
                if env.collision_occurred: # Check if collision happened *during* monitoring
                    forced_executed_collision += 1
                    print(f"⚠️ Forced change ({monitor_target_action}) agreed but resulted in COLLISION.")
                else:
                    forced_executed_safe += 1
                    print(f"✅ Forced change ({monitor_target_action}) executed successfully.")
                monitoring_change = False # Stop monitoring this attempt
            elif monitor_steps_left <= 0:
                print(f"❌ Forced change ({monitor_target_action}) agreed but timed out (not executed).")
                monitoring_change = False # Timed out
            elif env.collision_occurred: # Collision happened before completion
                 forced_executed_collision += 1
                 print(f"⚠️ Forced change ({monitor_target_action}) agreed but resulted in COLLISION before completion.")
                 monitoring_change = False

        # Trigger a new forced attempt if not currently monitoring one
        if not monitoring_change and step_count > 0 and step_count % eval_config.FORCE_CHANGE_INTERVAL_STEPS == 0:
            if can_go_left and can_go_right:
                target_action = random.choice([1, 2]) # Choose randomly if both possible
            elif can_go_left:
                target_action = 1 # Target left
            elif can_go_right:
                target_action = 2 # Target right

            if target_action != -1:
                forced_attempts += 1
                print(f"\n--- Step {step_count}: Triggering Forced Lane Change Attempt (Target Action: {target_action}) ---")


        # 3. Get Model Action
        if model_type == 'dqn':
            action = get_dqn_action(model, state_norm, current_lane_idx, config, device)
        elif model_type == 'ppo':
            action = get_ppo_action(model, state_norm, current_lane_idx, device)
        else:
            action = 0 # Fallback

        # 4. Handle Forced Change Logic
        if target_action != -1:
            print(f"   - Model chose action: {action}")
            if action == target_action:
                forced_agreed += 1
                print(f"   - Model AGREED with forced action {target_action}. Executing and monitoring...")
                monitoring_change = True
                monitor_steps_left = eval_config.FORCE_CHANGE_MONITOR_STEPS
                monitor_target_action = target_action
                monitor_start_lane = current_lane_idx
                monitor_start_pos = env.get_vehicle_info()['pos']
            else:
                print(f"   - Model DISAGREED (chose {action}). Executing model's choice, not monitoring.")
                # Continue with the model's chosen action, don't monitor

        # Count model-initiated lane changes (only if not monitoring a successful forced one)
        if action != 0 and not monitoring_change:
             model_lane_changes += 1

        # 5. Step Environment
        next_state_raw, done = env.step(action)

        # Update state and metrics
        state_raw = next_state_raw
        step_count += 1
        vehicle_info = env.get_vehicle_info()
        speeds.append(vehicle_info['speed'])

        # If a collision occurs, stop monitoring any forced change immediately
        if env.collision_occurred:
            done = True # Ensure loop terminates
            if monitoring_change:
                 forced_executed_collision += 1 # Assume collision was related
                 print(f"⚠️ Forced change ({monitor_target_action}) monitoring interrupted by COLLISION.")
                 monitoring_change = False


    # Episode End
    avg_speed = np.mean(speeds) if speeds else 0.0
    total_dist = env.get_vehicle_info()['dist']
    collided = env.collision_occurred

    # Handle case where monitoring was ongoing at episode end (timeout)
    if monitoring_change:
        print(f"❌ Forced change ({monitor_target_action}) monitoring ongoing at episode end (timeout).")

    result = EpisodeResult(
        steps=step_count,
        collided=collided,
        avg_speed=avg_speed,
        total_dist=total_dist,
        forced_attempts=forced_attempts,
        forced_agreed=forced_agreed,
        forced_executed_safe=forced_executed_safe,
        forced_executed_collision=forced_executed_collision,
        model_lane_changes=model_lane_changes
    )

    # Return result and the final state of the normalizer for this episode run
    return result, current_obs_normalizer


#####################################
# Main Evaluation Script            #
#####################################
def main():
    eval_config = EvalConfig()
    dqn_train_config = DQN_Config()
    ppo_train_config = PPO_Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- File Checks ---
    if not os.path.exists(eval_config.DQN_MODEL_PATH):
        print(f"ERROR: DQN Model path not found: {eval_config.DQN_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(eval_config.PPO_MODEL_PATH):
        print(f"ERROR: PPO Model path not found: {eval_config.PPO_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(eval_config.EVAL_SUMO_CONFIG):
        print(f"ERROR: Evaluation SUMO config not found: {eval_config.EVAL_SUMO_CONFIG}")
        # Attempt to find referenced files
        try:
            with open(eval_config.EVAL_SUMO_CONFIG, 'r') as f:
                content = f.read()
                if 'new.net.xml' not in content: print("Warning: 'new.net.xml' not mentioned in sumocfg?")
                if 'new.rou.xml' not in content: print("Warning: 'new.rou.xml' not mentioned in sumocfg?")
            if not os.path.exists("new.net.xml"): print("Warning: new.net.xml not found.")
            if not os.path.exists("new.rou.xml"): print("Warning: new.rou.xml not found.")
        except Exception as e:
            print(f"Could not read SUMO config: {e}")
        sys.exit(1)


    # --- Load Models ---
    print("\n--- Loading Models ---")
    dqn_model = load_model(eval_config.DQN_MODEL_PATH, 'dqn', dqn_train_config, device)
    ppo_model = load_model(eval_config.PPO_MODEL_PATH, 'ppo', ppo_train_config, device)

    # --- Initialize Normalizers ---
    # Create initial normalizer instances based on training configs
    # These will be copied and updated per-episode during evaluation runs
    dqn_obs_normalizer_init = RunningMeanStd(shape=(dqn_train_config.state_dim,), alpha=dqn_train_config.norm_update_rate) if dqn_train_config.normalize_observations else None
    ppo_obs_normalizer_init = RunningMeanStd(shape=(ppo_train_config.state_dim,), alpha=ppo_train_config.norm_update_rate) if ppo_train_config.normalize_observations else None
    print("Initialized normalizers (will be updated per-episode during eval).")


    # --- Initialize Environment ---
    # Pass a base seed, it will be incremented per episode pair
    base_seed = eval_config.EVAL_SEED
    env = EvaluationEnv(eval_config, base_seed) # Seed will be updated inside loop

    # --- Run Evaluation ---
    print(f"\n--- Running Evaluation ({eval_config.EVAL_EPISODES} Episodes per Model) ---")
    dqn_results_list: List[EpisodeResult] = []
    ppo_results_list: List[EpisodeResult] = []

    for i in tqdm(range(eval_config.EVAL_EPISODES), desc="Evaluating Episodes"):
        episode_seed = base_seed + i
        print(f"\n--- Episode {i+1}/{eval_config.EVAL_EPISODES} (Seed: {episode_seed}) ---")

        # --- Evaluate DQN ---
        print("Evaluating DQN...")
        env.sumo_seed = episode_seed # Set seed for this episode
        try:
            dqn_result, _ = evaluate_episode(
                dqn_model, 'dqn', env, dqn_train_config, dqn_obs_normalizer_init, device, eval_config
            )
            dqn_results_list.append(dqn_result)
            print(f"DQN Ep {i+1} Result: Steps={dqn_result.steps}, Collided={dqn_result.collided}, AvgSpeed={dqn_result.avg_speed:.2f}")
            print(f"  Forced Changes: Attempts={dqn_result.forced_attempts}, Agreed={dqn_result.forced_agreed}, ExecutedSafe={dqn_result.forced_executed_safe}, ExecutedCollision={dqn_result.forced_executed_collision}")
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
            print(f"Error during DQN evaluation episode {i+1}: {e}")
            if isinstance(e, KeyboardInterrupt): break
            env._close() # Ensure cleanup
            time.sleep(1)
        except Exception as e_other:
            print(f"Unexpected Error during DQN evaluation episode {i+1}: {e_other}")
            traceback.print_exc()
            env._close(); time.sleep(1)


        # --- Evaluate PPO ---
        print("\nEvaluating PPO...")
        env.sumo_seed = episode_seed # Use the SAME seed for PPO for fairness
        try:
            ppo_result, _ = evaluate_episode(
                ppo_model, 'ppo', env, ppo_train_config, ppo_obs_normalizer_init, device, eval_config
            )
            ppo_results_list.append(ppo_result)
            print(f"PPO Ep {i+1} Result: Steps={ppo_result.steps}, Collided={ppo_result.collided}, AvgSpeed={ppo_result.avg_speed:.2f}")
            print(f"  Forced Changes: Attempts={ppo_result.forced_attempts}, Agreed={ppo_result.forced_agreed}, ExecutedSafe={ppo_result.forced_executed_safe}, ExecutedCollision={ppo_result.forced_executed_collision}")
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
            print(f"Error during PPO evaluation episode {i+1}: {e}")
            if isinstance(e, KeyboardInterrupt): break
            env._close() # Ensure cleanup
            time.sleep(1)
        except Exception as e_other:
            print(f"Unexpected Error during PPO evaluation episode {i+1}: {e_other}")
            traceback.print_exc()
            env._close(); time.sleep(1)


    # Close env after all episodes
    env._close()
    print("\n--- Evaluation Finished ---")

    # --- Aggregate and Compare Results ---
    if not dqn_results_list or not ppo_results_list:
        print("No evaluation results collected. Exiting.")
        return

    results = {'dqn': {}, 'ppo': {}}
    for model_key, results_list in [('dqn', dqn_results_list), ('ppo', ppo_results_list)]:
        total_episodes = len(results_list)
        results[model_key]['total_episodes'] = total_episodes

        # Basic metrics
        results[model_key]['avg_steps'] = np.mean([r.steps for r in results_list])
        results[model_key]['std_steps'] = np.std([r.steps for r in results_list])
        results[model_key]['collision_rate'] = np.mean([r.collided for r in results_list]) * 100
        results[model_key]['avg_speed'] = np.mean([r.avg_speed for r in results_list])
        results[model_key]['std_speed'] = np.std([r.avg_speed for r in results_list])
        results[model_key]['avg_dist'] = np.mean([r.total_dist for r in results_list])
        results[model_key]['avg_model_lc'] = np.mean([r.model_lane_changes for r in results_list]) # Avg model-initiated changes

        # Forced change metrics
        total_forced_attempts = sum(r.forced_attempts for r in results_list)
        total_forced_agreed = sum(r.forced_agreed for r in results_list)
        total_forced_executed_safe = sum(r.forced_executed_safe for r in results_list)
        total_forced_executed_collision = sum(r.forced_executed_collision for r in results_list)

        results[model_key]['total_forced_attempts'] = total_forced_attempts
        # Agreement Rate: % of forced attempts where model chose the target action
        results[model_key]['forced_agreement_rate'] = (total_forced_agreed / total_forced_attempts * 100) if total_forced_attempts > 0 else 0
        # Execution Success Rate: % of *agreed* attempts that completed safely
        results[model_key]['forced_execution_success_rate'] = (total_forced_executed_safe / total_forced_agreed * 100) if total_forced_agreed > 0 else 0
        # Execution Collision Rate: % of *agreed* attempts that resulted in collision (during monitoring)
        results[model_key]['forced_execution_collision_rate'] = (total_forced_executed_collision / total_forced_agreed * 100) if total_forced_agreed > 0 else 0

    # --- Print Comparison ---
    print("\n--- Results Comparison ---")
    print(f"{'Metric':<32} | {'DQN':<20} | {'PPO':<20}")
    print("-" * 78)
    metrics_to_print = [
        ('Avg. Steps', 'avg_steps', '.1f'),
        ('Std Dev Steps', 'std_steps', '.1f'),
        ('Avg. Speed (m/s)', 'avg_speed', '.2f'),
        ('Std Dev Speed (m/s)', 'std_speed', '.2f'),
        ('Avg. Distance (m)', 'avg_dist', '.1f'),
        ('Collision Rate (%)', 'collision_rate', '.1f'),
        ('Avg. Model Lane Changes', 'avg_model_lc', '.1f'),
        ('Forced Attempts (Total)', 'total_forced_attempts', 'd'),
        ('Forced Agreement Rate (%)', 'forced_agreement_rate', '.1f'),
        ('Forced Execution Success Rate (%)', 'forced_execution_success_rate', '.1f'),
        ('Forced Exec. Collision Rate (%)', 'forced_execution_collision_rate', '.1f'),
    ]
    for name, key, fmt in metrics_to_print:
        dqn_val = results['dqn'].get(key, 'N/A')
        ppo_val = results['ppo'].get(key, 'N/A')
        dqn_str = format(dqn_val, fmt) if isinstance(dqn_val, (int, float)) else str(dqn_val)
        ppo_str = format(ppo_val, fmt) if isinstance(ppo_val, (int, float)) else str(ppo_val)
        print(f"{name:<32} | {dqn_str:<20} | {ppo_str:<20}")
    print("-" * 78)


    # --- Generate Plots ---
    print("\n--- 生成图表 ---") # Changed
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"evaluation_comparison_{timestamp}.png"
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle('模型评估对比 (DQN vs PPO)', fontsize=16) # Changed
    models = ['DQN', 'PPO'] # Keep algorithm names in English

    # Plot 1: Forced Change Agreement Rate
    rates = [results['dqn']['forced_agreement_rate'], results['ppo']['forced_agreement_rate']]
    axes[0, 0].bar(models, rates, color=['skyblue', 'lightcoral'])
    axes[0, 0].set_ylabel('比率 (%)') # Changed
    axes[0, 0].set_title('强制换道: 模型同意率') # Changed
    axes[0, 0].set_ylim(0, 105)
    for i, v in enumerate(rates): axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha='center')

    # Plot 2: Forced Change Execution Success Rate (of agreed attempts)
    rates = [results['dqn']['forced_execution_success_rate'], results['ppo']['forced_execution_success_rate']]
    axes[0, 1].bar(models, rates, color=['skyblue', 'lightcoral'])
    axes[0, 1].set_ylabel('比率 (%)') # Changed
    axes[0, 1].set_title('强制换道: 执行成功率\n(同意换道的百分比)') # Changed
    axes[0, 1].set_ylim(0, 105)
    for i, v in enumerate(rates): axes[0, 1].text(i, v + 1, f"{v:.1f}%", ha='center')

    # Plot 3: Overall Collision Rate
    rates = [results['dqn']['collision_rate'], results['ppo']['collision_rate']]
    axes[1, 0].bar(models, rates, color=['skyblue', 'lightcoral'])
    axes[1, 0].set_ylabel('比率 (%)') # Changed
    axes[1, 0].set_title('总碰撞率') # Changed
    axes[1, 0].set_ylim(0, max(rates + [5])) # Adjust ylim dynamically
    for i, v in enumerate(rates): axes[1, 0].text(i, v + 0.5, f"{v:.1f}%", ha='center')

    # Plot 4: Average Speed
    speeds_mean = [results['dqn']['avg_speed'], results['ppo']['avg_speed']]
    speeds_std = [results['dqn']['std_speed'], results['ppo']['std_speed']]
    axes[1, 1].bar(models, speeds_mean, yerr=speeds_std, capsize=5, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_ylabel('速度 (米/秒)') # Changed
    axes[1, 1].set_title('平均速度') # Changed
    for i, v in enumerate(speeds_mean): axes[1, 1].text(i, v + 0.5, f"{v:.2f}", ha='center')

    # Plot 5: Average Steps per Episode
    steps_mean = [results['dqn']['avg_steps'], results['ppo']['avg_steps']]
    steps_std = [results['dqn']['std_steps'], results['ppo']['std_steps']]
    axes[2, 0].bar(models, steps_mean, yerr=steps_std, capsize=5, color=['skyblue', 'lightcoral'])
    axes[2, 0].set_ylabel('步数') # Changed
    axes[2, 0].set_title('每回合平均步数') # Changed
    for i, v in enumerate(steps_mean): axes[2, 0].text(i, v + 5, f"{v:.1f}", ha='center')

    # Plot 6: Average Model-Initiated Lane Changes
    lc_mean = [results['dqn']['avg_model_lc'], results['ppo']['avg_model_lc']]
    axes[2, 1].bar(models, lc_mean, color=['skyblue', 'lightcoral'])
    axes[2, 1].set_ylabel('次数') # Changed
    axes[2, 1].set_title('模型发起的平均换道次数') # Changed
    for i, v in enumerate(lc_mean): axes[2, 1].text(i, v + 0.2, f"{v:.1f}", ha='center')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(plot_filename)
    print(f"对比图表已保存至: {plot_filename}") # Changed
    # plt.show() # Optionally show plot interactively


    # --- Save Data ---
    data_filename = f"evaluation_comparison_data_{timestamp}.json"
    # Add raw episode data for potential deeper analysis
    results['dqn']['raw_results'] = [r._asdict() for r in dqn_results_list]
    results['ppo']['raw_results'] = [r._asdict() for r in ppo_results_list]
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            # Custom encoder for numpy types if any creep in (shouldn't with namedtuple)
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        print(f"Comparison data saved to: {data_filename}")
    except Exception as e:
        print(f"Error saving comparison data: {e}")

if __name__ == "__main__":
    main()