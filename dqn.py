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
import torch.optim as optim
import torch.nn.functional as F # For loss function
# from torch.distributions import Categorical # Not used
import traci
from tqdm import tqdm
import matplotlib.pyplot as plt
import random # For PER sampling
from collections import deque, namedtuple # For Replay Buffer and N-step
from typing import List, Tuple, Dict, Optional, Any
import socket # 用于端口检查
import traceback # 用于打印详细错误
import collections # For deque in normalization and replay buffer
import math # For PER beta annealing, Noisy Nets

# Solve Chinese garbled characters in matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

#####################
#     Config Area   #
#####################
class Config:
    # --- SUMO Config --- (Keep same as PPO)
    sumo_binary = "sumo" # or "sumo-gui"
    config_path = "a.sumocfg"
    step_length = 0.2
    ego_vehicle_id = "drl_ego_car"
    ego_type_id = "car_ego"
    port_range = (8890, 8900)

    # --- Behavior Cloning (BC) ---
    use_bc = False # Disabled for DQN

    # --- DQN Training ---
    dqn_episodes = 500 # Keep same number of episodes as PPO for comparison
    max_steps = 8000   # Keep same step limit per episode
    log_interval = 10
    save_interval = 50

    # --- DQN Hyperparameters ---
    gamma = 0.99 # Discount factor (Same as PPO)
    initial_learning_rate = 7e-5 # Potentially lower LR for C51/Noisy (was 1e-4)
    batch_size = 512 # Batch size for sampling from replay buffer (Same as PPO)
    hidden_size = 256 # Network hidden layer size (Same as PPO)
    replay_buffer_size = 100000 # Size of the experience replay buffer
    target_update_freq = 2500   # How many training STEPS between target network updates (was 1000)
    learning_starts = 7500      # Number of steps to collect before starting training (was 5000)
    use_double_dqn = True       # Use Double DQN improvement (applied within C51 logic)

    # --- Epsilon-Greedy Exploration (REMOVED - Using Noisy Nets) ---
    # epsilon_start = 1.0
    # epsilon_end = 0.05
    # epsilon_decay_total_steps = int(dqn_episodes * 300 * 0.7)

    # --- Noisy Networks ---
    use_noisy_nets = True       # Enable Noisy Networks for exploration
    noisy_sigma_init = 0.5     # Initial std deviation for NoisyLinear layers

    # --- Distributional DQN (C51) ---
    use_distributional = True   # Enable C51 Distributional DQN
    v_min = -110                # Minimum possible return value (adjust based on rewards: ~-100 collision + some negative step rewards)
    v_max = 30                  # Maximum possible return value (Increased slightly due to reward changes)
    num_atoms = 51              # Number of atoms in the distribution support

    # --- Normalization --- (Keep same as PPO)
    normalize_observations = True # Enable observation normalization
    normalize_rewards = True      # Enable reward normalization (scaling) - Applied *before* N-step calc now
    obs_norm_clip = 5.0           # Clip normalized observations to [-5, 5]
    reward_norm_clip = 10.0         # Clip normalized rewards to [-10, 10]
    norm_update_rate = 0.001      # Rate for updating running mean/std (ema alpha)

    # --- Prioritized Experience Replay (PER) ---
    use_per = True
    per_alpha = 0.6             # Prioritization exponent (0=uniform, 1=fully prioritized)
    per_beta_start = 0.4        # Initial IS weight exponent (0=no correction, 1=full correction)
    per_beta_end = 1.0          # Final IS weight exponent
    # Anneal beta over training duration (estimate steps better if needed)
    # Keep original estimate, may need tuning based on actual steps/episode
    per_beta_annealing_steps = int(dqn_episodes * 300 * 0.8) # Anneal over 80% of estimated total steps
    per_epsilon = 1e-5          # Small value added to priorities

    # --- N-Step Returns ---
    use_n_step = True
    n_step = 5                  # Number of steps for N-step returns (was 3)

    # --- State/Action Space --- (Keep same as PPO)
    state_dim = 12
    action_dim = 3 # 0: Keep, 1: Left, 2: Right

    # --- Environment Parameters --- (Keep same as PPO)
    max_speed_global = 33.33 # m/s (~120 km/h)
    max_distance = 100.0     # m
    lane_max_speeds = [33.33, 27.78, 22.22] # m/s - Must match a.net.xml

    # --- Reward Function Parameters (REVISED - Encouraging Lane Changes) ---
    reward_collision = -100.0 # Keeping original collision penalty
    reward_high_speed_scale = 0.25 # Increased (was 0.15)
    reward_low_speed_penalty_scale = 0.1
    reward_lane_change_penalty = -0.02 # Reduced penalty further (was -0.05)
    reward_faster_lane_bonus = 0.6 # *** NEW: Increased bonus for moving to a faster potential lane *** (was 0.1)
    reward_staying_slow_penalty_scale = 0.1 # *** NEW: Penalty for not changing to faster lane ***
    time_alive_reward = 0.01
    reward_comfort_penalty_scale = 0.05
    target_speed_factor = 0.95
    safe_distance_penalty_scale = 0.2
    min_buffer_dist_reward = 5.0
    time_gap_reward = 0.8
    # Minimum safe distance for considering a lane change in reward calc
    min_safe_change_dist = 15.0

#####################
#   Normalization    #
#####################
# RunningMeanStd remains EXACTLY the same
class RunningMeanStd:
    """Calculates the running mean and standard deviation"""
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4, alpha: float = 0.001):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.alpha = alpha # Exponential Moving Average alpha

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        # EMA update
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1.0 - self.alpha) * self.var + self.alpha * batch_var
        self.count += batch_count # Keep count for reference, but EMA drives the update

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8) # Add epsilon here too for safety

# Reward scaling helper - NOW NORMALIZES SINGLE STEP REWARDS
class RewardNormalizer:
    def __init__(self, gamma: float, epsilon: float = 1e-8, alpha: float = 0.001, clip: float = 10.0):
        self.ret_rms = RunningMeanStd(shape=(), alpha=alpha) # Use RunningMeanStd for returns
        self.epsilon = epsilon
        self.clip = clip
        # Gamma is not directly used here anymore, but kept for signature consistency

    def normalize(self, r: np.ndarray) -> np.ndarray:
        # Update running stats with the current reward(s)
        self.ret_rms.update(r) # Assuming r is a single reward or batch of rewards
        # Normalize: (r - mean) / std, but usually just r / std is used for rewards
        norm_r = r / self.ret_rms.std
        return np.clip(norm_r, -self.clip, self.clip)


#####################
#   Utility Functions #
#####################
# get_available_port, kill_sumo_processes remain EXACTLY the same
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
    # print("Attempting to terminate lingering SUMO processes...") # Less verbose
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
        # if killed: print("Terminated one or more SUMO processes.")
    except Exception as e: print(f"Warning: Error terminating SUMO processes: {e}")
    time.sleep(0.1) # Shorter sleep

# Linear decay function (used for PER beta)
def linear_decay(start_val, end_val, total_steps, current_step):
    """Linear decay calculation"""
    if current_step >= total_steps:
        return end_val
    return start_val + (end_val - start_val) * (current_step / total_steps)

#####################
#   SUMO Environment Wrapper #
#####################
# Modified slightly for obs normalization storage and reward calculation tweak
class SumoEnv:
    def __init__(self, config: Config):
        self.config = config
        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_speed = 0.0 # Used for comfort penalty
        self.last_raw_state = np.zeros(config.state_dim) # Store raw state before norm
        self.last_norm_state = np.zeros(config.state_dim) # Store normalized state
        self.last_lane_idx = 0 # Store last lane index for reward calc

        # Normalization (applied within env)
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate, clip=config.reward_norm_clip) if config.normalize_rewards else None

        # Metrics
        self.reset_metrics()

    def reset_metrics(self):
        """Reset episode metrics"""
        self.change_lane_count = 0
        self.collision_occurred = False
        self.current_step = 0
        self.last_action = 0 # Last executed action

    def _start_sumo(self):
        """Start SUMO instance and connect TraCI"""
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.config.port_range[0], self.config.port_range[1])
        except IOError as e:
             print(f"ERROR: Failed to find available port: {e}")
             sys.exit(1)

        sumo_cmd = [
            self.config.sumo_binary, "-c", self.config.config_path,
            "--remote-port", str(self.traci_port),
            "--step-length", str(self.config.step_length),
            "--collision.check-junctions", "true",
            "--collision.action", "warn", # Use "warn" to detect collisions without stopping simulation instantly
            "--time-to-teleport", "-1", # Disable teleporting
            "--no-warnings", "true",
            "--seed", str(np.random.randint(0, 10000))
        ]
        if self.config.sumo_binary == "sumo-gui":
             sumo_cmd.extend(["--quit-on-end", "true"]) # Close GUI automatically

        try:
             # Redirect stdout/stderr only for non-GUI mode to avoid suppressing GUI errors
             stdout_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
        except FileNotFoundError:
             print(f"ERROR: SUMO executable '{self.config.sumo_binary}' not found.")
             sys.exit(1)
        except Exception as e:
             print(f"ERROR: Failed to start SUMO process: {e}")
             sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5) # Slightly adjusted delay
                traci.init(self.traci_port)
                # print(f"✅ SUMO TraCI connected (Port: {self.traci_port}).") # Less verbose
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
        """Add the ego vehicle to the simulation"""
        ego_route_id = "route_E0"
        if ego_route_id not in traci.route.getIDList():
             try: traci.route.add(ego_route_id, ["E0"])
             except traci.exceptions.TraCIException as e: raise RuntimeError(f"Failed to add route '{ego_route_id}': {e}")

        if self.config.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.config.ego_type_id)
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0") # Red
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0") # Make it strategically flexible
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcSpeedGain", "2.0") # Be more willing to change for speed (increased from 1.0)
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcCooperative", "0.5") # Slightly less cooperative to prioritize own gain
                traci.vehicletype.setParameter(self.config.ego_type_id, "jmIgnoreFoeProb", "0.1") # Small chance to ignore leader when merging/changing lanes if blocked
                traci.vehicletype.setParameter(self.config.ego_type_id, "carFollowModel", "IDM") # Use standard IDM model
                traci.vehicletype.setParameter(self.config.ego_type_id, "minGap", "2.5") # Standard min gap
            except traci.exceptions.TraCIException as e: print(f"Warning: Failed to set parameters for Ego type '{self.config.ego_type_id}': {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.config.ego_vehicle_id)
                time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"Warning: Failed to remove residual Ego: {e}")

        try:
            # Try adding on a specific lane first, fallback to random
            start_lane = random.choice([0, 1, 2])
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                              typeID=self.config.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max") # Try specific lane

            wait_steps = int(2.0 / self.config.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                     ego_appeared = True
                     break
            if not ego_appeared:
                 # Fallback if specific lane failed (e.g., blocked)
                 print(f"Warning: Failed to add Ego on lane {start_lane}, trying random lane.")
                 traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                                   typeID=self.config.ego_type_id, depart="now",
                                   departLane="random", departSpeed="max")
                 for _ in range(wait_steps):
                     traci.simulationStep()
                     if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                         ego_appeared = True
                         break
                 if not ego_appeared:
                    raise RuntimeError(f"Ego vehicle did not appear within {wait_steps*2} steps.")

        except traci.exceptions.TraCIException as e:
            print(f"ERROR: Failed to add Ego vehicle '{self.config.ego_vehicle_id}': {e}")
            raise RuntimeError("Failed adding Ego vehicle.")


    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode, return NORMALIZED state"""
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.reset_metrics()

        self.last_speed = 0.0
        self.last_raw_state = np.zeros(self.config.state_dim)
        self.last_norm_state = np.zeros(self.config.state_dim)
        self.last_lane_idx = 0 # Reset last lane
        norm_state = np.zeros(self.config.state_dim)

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
             try:
                 self.last_speed = traci.vehicle.getSpeed(self.config.ego_vehicle_id)
                 self.last_lane_idx = traci.vehicle.getLaneIndex(self.config.ego_vehicle_id)
                 # Get raw state first
                 raw_state = self._get_raw_state()
                 self.last_raw_state = raw_state.copy()
                 # Then normalize it
                 norm_state = self._normalize_state(raw_state)
                 self.last_norm_state = norm_state.copy()
             except traci.exceptions.TraCIException:
                 print("Warning: TraCI exception during initial state fetch in reset.")
                 pass # Keep defaults if error
        else:
             print("Warning: Ego vehicle not found immediately after add/wait in reset.")

        return norm_state # Return the NORMALIZED state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize the state using running mean/std."""
        if self.obs_normalizer:
            self.obs_normalizer.update(state.reshape(1, -1)) # Update with single observation
            norm_state = (state - self.obs_normalizer.mean) / self.obs_normalizer.std
            norm_state = np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip)
            return norm_state.astype(np.float32)
        else:
            return state # Return raw state if normalization is off


    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """Get distance and relative speed of nearest vehicles (unchanged)"""
        max_dist = self.config.max_distance
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        ego_road = traci.vehicle.getRoadID(ego_id)
        if ego_road == "" or not ego_road.startswith("E"): return infos # Check if on expected edge

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                # Check if the vehicle is on the same edge first
                veh_road = traci.vehicle.getRoadID(veh_id)
                if veh_road != ego_road: continue

                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                dx = veh_pos[0] - ego_pos[0] # Assuming driving along X-axis mostly
                dy = veh_pos[1] - ego_pos[1] # Lateral distance
                distance = math.sqrt(dx**2 + dy**2) # Euclidean distance

                if distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed # Positive if ego is faster

                # Check lane relative position more carefully
                if veh_lane == ego_lane: # Same lane
                    if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                    # Ignore behind vehicle in same lane for this state representation
                elif veh_lane == ego_lane - 1: # Left lane
                    if dx > -5 and distance < infos['left_front'][0]: # Slightly behind to front
                        infos['left_front'] = (distance, rel_speed)
                    elif dx <= -5 and distance < infos['left_back'][0]: # Further behind
                        infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1: # Right lane
                     if dx > -5 and distance < infos['right_front'][0]: # Slightly behind to front
                        infos['right_front'] = (distance, rel_speed)
                     elif dx <= -5 and distance < infos['right_back'][0]: # Further behind
                        infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue # Skip vehicle if TraCI error occurs
        return infos

    def _get_raw_state(self) -> np.ndarray:
        """Get the current environment state (RAW values before normalization)"""
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            return self.last_raw_state # Return last known raw state if ego vanished

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if current_road == "" or not current_road.startswith("E"):
                return self.last_raw_state # Not on the expected road segment

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            num_lanes = traci.edge.getLaneNumber("E0") # Get actual number of lanes

            # Ensure ego_lane is valid before using it for checks
            if not (0 <= ego_lane < num_lanes):
                 # Attempt to refetch or use last known valid lane
                 print(f"Warning: Invalid ego lane {ego_lane} detected. Attempting refetch...")
                 time.sleep(0.05)
                 if ego_id in traci.vehicle.getIDList():
                     ego_lane = traci.vehicle.getLaneIndex(ego_id)
                 if not (0 <= ego_lane < num_lanes):
                     print(f"Warning: Still invalid lane {ego_lane}. Using last valid lane {self.last_lane_idx}.")
                     ego_lane = self.last_lane_idx
                     if not (0 <= ego_lane < num_lanes): # Final fallback
                         ego_lane = 0

            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # --- Raw State Features ---
            state[0] = ego_speed
            state[1] = float(ego_lane) # Keep lane index as float for consistency/normalization
            state[2] = min(surround_info['front'][0], self.config.max_distance)
            state[3] = surround_info['front'][1] # Relative speed (not clipped here)
            state[4] = min(surround_info['left_front'][0], self.config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.config.max_distance)
            state[7] = min(surround_info['right_front'][0], self.config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.config.max_distance)
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"Warning: NaN or Inf detected in raw state calculation. Using last valid raw state.")
                return self.last_raw_state

            self.last_raw_state = state.copy() # Store the latest valid raw state

        except traci.exceptions.TraCIException as e:
            # Check if it's a "vehicle not found" error, which might be expected near end/collision
            if "Vehicle '" + ego_id + "' is not known" in str(e):
                 pass # Expected if vehicle leaves simulation or collides
            else:
                 print(f"Warning: TraCI error getting raw state for {ego_id}: {e}. Returning last known raw state.")
            return self.last_raw_state
        except Exception as e:
            print(f"Warning: Unknown error getting raw state for {ego_id}: {e}. Returning last known raw state.")
            traceback.print_exc()
            return self.last_raw_state

        return state # Return raw state


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """Execute an action, return (next_normalized_state, normalized_reward, done, next_lane_index)"""
        done = False
        raw_reward = 0.0 # Reward BEFORE normalization
        next_lane_index = self.last_lane_idx # Initialize with current lane
        ego_id = self.config.ego_vehicle_id
        self.last_action = action # Store the action taken

        if ego_id not in traci.vehicle.getIDList():
             # Return last known normalized state, collision reward, done=True, last known lane index
             return self.last_norm_state, self.reward_normalizer.normalize(np.array([self.config.reward_collision]))[0] if self.reward_normalizer else self.config.reward_collision, True, self.last_lane_idx

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id)
            current_road = traci.vehicle.getRoadID(ego_id)
            num_lanes = traci.edge.getLaneNumber(current_road) if current_road.startswith("E") else 0

             # Ensure current_lane is valid before action execution
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)


            # 1. Execute Action if valid
            lane_change_requested = False
            if action == 1 and current_lane > 0: # Try Left
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0) # Request change
                lane_change_requested = True
            elif action == 2 and num_lanes > 0 and current_lane < (num_lanes - 1): # Try Right
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0) # Request change
                lane_change_requested = True

            # 2. Simulation Step
            traci.simulationStep()
            self.current_step += 1

            # 3. Check Status & Calculate Reward AFTER step
            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True
                done = True
                raw_reward = self.config.reward_collision
                next_norm_state = self.last_norm_state # Return last known normalized state
                next_lane_index = self.last_lane_idx # Use last known lane
                normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward
                return next_norm_state, normalized_reward, done, next_lane_index

            # Explicit collision check
            collisions = traci.simulation.getCollisions()
            ego_collided = False
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True
                    ego_collided = True
                    done = True
                    raw_reward = self.config.reward_collision
                    break

            # Get next state (raw first, then normalize)
            next_raw_state = self._get_raw_state()
            next_norm_state = self._normalize_state(next_raw_state)
            next_lane_index = int(round(next_raw_state[1])) # Get lane index AFTER step

            # Calculate reward based on the state *after* the step, unless collision happened
            if not self.collision_occurred:
                 # Extract necessary info from the *RAW* state for reward calc
                 current_speed_after_step = next_raw_state[0]
                 current_lane_after_step = int(round(next_raw_state[1])) # = next_lane_index
                 front_dist_after_step = next_raw_state[2]
                 left_front_dist = next_raw_state[4]
                 right_front_dist = next_raw_state[7]
                 can_change_left_after_step = next_raw_state[10] > 0.5
                 can_change_right_after_step = next_raw_state[11] > 0.5


                 # Check if lane change actually occurred
                 actual_lane_change = (self.last_lane_idx != current_lane_after_step)
                 if actual_lane_change:
                     self.change_lane_count += 1
                 # Apply penalty only if a change was *requested* in this step
                 effective_action = action if lane_change_requested else 0

                 # Pass necessary state components to reward function
                 raw_reward = self._calculate_reward(effective_action, current_speed_after_step,
                                                 current_lane_after_step, front_dist_after_step,
                                                 self.last_lane_idx, # Pass previous lane
                                                 can_change_left_after_step, can_change_right_after_step,
                                                 left_front_dist, right_front_dist)

            # 4. Check other termination conditions
            if traci.simulation.getTime() >= 3600: done = True
            if self.current_step >= self.config.max_steps: done = True

        except traci.exceptions.TraCIException as e:
            if ego_id not in traci.vehicle.getIDList(): self.collision_occurred = True
            raw_reward = self.config.reward_collision # Penalize for error/collision
            done = True
            next_norm_state = self.last_norm_state
            next_lane_index = self.last_lane_idx
        except Exception as e:
            print(f"ERROR: Unknown exception during step {self.current_step}: {e}")
            traceback.print_exc()
            done = True
            raw_reward = self.config.reward_collision
            self.collision_occurred = True # Assume collision on unknown error
            next_norm_state = self.last_norm_state
            next_lane_index = self.last_lane_idx

        # 5. Update last state info for next step
        self.last_speed = next_raw_state[0]
        self.last_lane_idx = next_lane_index # Update last lane index based on state AFTER step
        self.last_norm_state = next_norm_state.copy() # Store normalized state

        # *** Normalize reward BEFORE returning ***
        normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward

        # Return NORMALIZED state, NORMALIZED reward, done flag, and NEXT lane index
        return next_norm_state, normalized_reward, done, next_lane_index


    def _calculate_reward(self, action_taken: int, current_speed: float, current_lane: int,
                           front_dist: float, previous_lane: int,
                           can_change_left: bool, can_change_right: bool,
                           left_front_dist: float, right_front_dist: float) -> float:
        """Calculate reward based on the state AFTER the step (Uses raw values), incorporating new penalties/bonuses."""
        if self.collision_occurred: return 0.0 # Reward already handled

        try:
            num_lanes = len(self.config.lane_max_speeds)
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)
            if not (0 <= previous_lane < num_lanes): # Ensure previous lane is valid too
                 previous_lane = current_lane # Default to current if invalid

            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            # --- Reward Components ---
            # Speed Reward / Penalty (Increased Scale)
            speed_diff = abs(current_speed - target_speed)
            speed_reward = np.exp(- (speed_diff / (target_speed * 0.3 + 1e-6))**2 ) * self.config.reward_high_speed_scale

            # Low Speed Penalty
            low_speed_penalty = 0.0
            low_speed_threshold = target_speed * 0.6
            if current_speed < low_speed_threshold:
                low_speed_penalty = (current_speed / low_speed_threshold - 1.0) * self.config.reward_low_speed_penalty_scale

            # Lane Change Penalty & Bonus (Reduced Penalty, Increased Bonus)
            lane_change_penalty = 0.0
            lane_change_bonus = 0.0
            if action_taken != 0: # If a lane change was attempted
                lane_change_penalty = self.config.reward_lane_change_penalty
                # Check if the change resulted in moving to a potentially faster lane
                # Assumes lower index = faster lane (as per config.lane_max_speeds)
                if current_lane < previous_lane: # Moved left to a faster potential lane
                     lane_change_bonus = self.config.reward_faster_lane_bonus
                # Could add penalty for moving right unnecessarily, but let's keep it simple

            # *** NEW: Penalty for Staying in Slower Lane ***
            staying_slow_penalty = 0.0
            if action_taken == 0: # Only penalize if agent chose 'Keep Lane'
                # Check left lane (faster)
                left_lane_idx = current_lane - 1
                if left_lane_idx >= 0 and self.config.lane_max_speeds[left_lane_idx] > lane_max_speed:
                    if can_change_left and left_front_dist > self.config.min_safe_change_dist:
                        # Faster lane available to left, change possible, and front is clear enough
                        staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale
                # Check right lane (faster - less common scenario but possible if misconfigured)
                # Only apply penalty if not already penalized for left option
                elif staying_slow_penalty == 0.0:
                    right_lane_idx = current_lane + 1
                    if right_lane_idx < num_lanes and self.config.lane_max_speeds[right_lane_idx] > lane_max_speed:
                         if can_change_right and right_front_dist > self.config.min_safe_change_dist:
                             # Faster lane available to right, change possible, and front is clear enough
                             staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale


            # Safety Distance Penalty
            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < self.config.max_distance:
                if front_dist < min_safe_dist:
                     safety_dist_penalty = max(-1.0, (front_dist / min_safe_dist - 1.0)) * self.config.safe_distance_penalty_scale

            # Comfort Penalty
            comfort_penalty = 0.0
            acceleration = (current_speed - self.last_speed) / self.config.step_length
            harsh_braking_threshold = -3.0
            if acceleration < harsh_braking_threshold:
                 comfort_penalty = (acceleration - harsh_braking_threshold) * self.config.reward_comfort_penalty_scale

            # Time Alive Reward
            time_alive = self.config.time_alive_reward

            # --- Total Reward ---
            total_reward = (speed_reward +
                            low_speed_penalty +
                            lane_change_penalty +
                            lane_change_bonus +
                            staying_slow_penalty + # Add new penalty
                            safety_dist_penalty +
                            comfort_penalty +
                            time_alive)

            return total_reward

        except Exception as e:
            print(f"Warning: Error calculating reward: {e}. Returning 0.")
            traceback.print_exc()
            return 0.0

    def _close(self):
        """Close SUMO instance and TraCI connection (Exact same as ppo.py)"""
        if self.sumo_process:
            try:
                traci.close()
            except Exception as e: pass
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


#####################
#   DQN Components   #
#####################

# --- N-Step Transition (Added next_lane_index) ---
NStepTransition = namedtuple('NStepTransition', ('state', 'action', 'reward', 'next_state', 'done', 'next_lane_index'))

# --- Experience for Replay Buffer (PER + N-Step) (Added next_lane_index) ---
Experience = namedtuple('Experience', ('state', 'action', 'n_step_reward', 'next_state', 'done', 'n', 'next_lane_index'))


# --- SumTree for Prioritized Replay --- (Unchanged)
class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left = 2 * idx + 1; right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s - self.tree[left])
    def total(self): return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0
        if self.n_entries < self.capacity: self.n_entries += 1
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        # Robustness check: Ensure data is valid before returning
        data = self.data[dataIdx]
        if not isinstance(data, Experience):
            print(f"Warning: Corrupted data found in SumTree at dataIdx {dataIdx} (tree idx {idx}). data: {data}. Attempting to return dummy or neighboring data.")
            # Option 1: Return dummy (might cause issues downstream)
            # return (idx, self.tree[idx], Experience(np.zeros(self.config.state_dim), 0, 0, np.zeros(self.config.state_dim), True, 1, 0))
            # Option 2: Try retrieving neighbor (less ideal)
            # Try sampling again with slightly different 's' - might be complex
            # Simplest recovery: Return the priority but indicate data is bad upstream
            # For now, let's return the potentially bad data and let the sampling loop handle retries
            pass # Allow the calling function (`sample`) to detect invalid type
        return (idx, self.tree[idx], data)
    def __len__(self): return self.n_entries

# --- Prioritized Replay Buffer (PER + N-Step) --- (Unchanged logic, relies on Experience tuple)
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, config: Config):
        self.config = config # Store config for potential use (e.g., state_dim in checks)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
    def push(self, experience: Experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    def sample(self, batch_size: int, beta: float) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        experiences = []; indices = np.empty((batch_size,), dtype=np.int32); is_weights = np.empty((batch_size,), dtype=np.float32)
        segment = self.tree.total() / batch_size
        if self.tree.total() == 0: # Avoid division by zero if buffer is empty/corrupt
             raise RuntimeError("SumTree total is zero, cannot sample.")

        for i in range(batch_size):
            attempt = 0
            while attempt < 5: # Retry sampling a few times if invalid data is hit
                a = segment * i; b = segment * (i + 1); s = random.uniform(a, b)
                # Ensure s does not exceed total (can happen with floating point issues near the end)
                s = min(s, self.tree.total() - 1e-6)
                try:
                    (idx, p, data) = self.tree.get(s)

                    # Critical check: Ensure 'data' is the correct type
                    if not isinstance(data, Experience):
                        print(f"Warning: Sampled invalid data (type: {type(data)}) at index {idx}, tree total {self.tree.total()}, segment [{a:.4f},{b:.4f}], s {s:.4f}. Retrying sample ({attempt+1}/5)...")
                        attempt += 1
                        time.sleep(0.01) # Small delay before retry
                        segment = self.tree.total() / batch_size # Recalculate segment in case total changed
                        continue # Retry the loop

                    # Check for None or other invalid content within Experience if necessary
                    # e.g., if data.state is None: ... continue

                    experiences.append(data); indices[i] = idx; prob = p / self.tree.total()
                    if prob <= 0: # Handle zero probability case
                         print(f"Warning: Sampled priority {p} resulted in zero probability (total: {self.tree.total()}). Setting small probability.")
                         prob = self.config.per_epsilon / self.tree.total() # Use epsilon

                    is_weights[i] = np.power(self.tree.n_entries * prob, -beta)
                    break # Successful sample, exit retry loop
                except Exception as e:
                    print(f"Error during SumTree.get or processing for s={s}, idx={idx}: {e}")
                    attempt += 1
                    time.sleep(0.01)
                    segment = self.tree.total() / batch_size # Recalculate segment

            if attempt == 5:
                 raise RuntimeError(f"Failed to sample valid Experience data from SumTree after 5 attempts. Tree total: {self.tree.total()}, n_entries: {self.tree.n_entries}")


        is_weights /= (is_weights.max() + 1e-8) # Normalize, add epsilon for stability
        return experiences, is_weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + self.config.per_epsilon # Use absolute TD error, add epsilon
        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < self.tree.capacity * 2 - 1):
                print(f"Warning: Invalid index {idx} provided to update_priorities. Skipping.")
                continue
            if not np.isfinite(priority):
                 print(f"Warning: Non-finite priority {priority} for index {idx}. Clamping.")
                 priority = self.max_priority # Clamp to max known priority

            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    def __len__(self) -> int: return self.tree.n_entries

# --- Noisy Linear Layer --- (Unchanged)
class NoisyLinear(nn.Module):
    """Noisy Linear Layer for Factorised Gaussian Noise"""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters for the weight mean and standard deviation
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        # Learnable parameters for the bias mean and standard deviation
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Factorised noise parameters (non-learnable buffers)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize weights and biases"""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features)) # Corrected: use out_features for bias sigma init

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size, device=self.weight_mu.device) # Ensure noise is on the correct device
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """Generate new noise vectors"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # Outer product to create factorised noise matrix
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out) # Bias noise is just the output noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights and biases"""
        if self.training: # Only apply noise during training
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else: # Use mean weights/biases during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


# --- Q-Network (Modified for C51 & Noisy Nets) --- (Unchanged Network Architecture)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, config: Config):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = config.num_atoms
        self.use_noisy = config.use_noisy_nets
        self.sigma_init = config.noisy_sigma_init # Pass sigma_init

        # Layer constructor based on config
        # Pass sigma_init if using NoisyLinear
        linear = lambda in_f, out_f: NoisyLinear(in_f, out_f, self.sigma_init) if self.use_noisy else nn.Linear(in_f, out_f)

        self.feature_layer = nn.Sequential(
            linear(state_dim, hidden_size), nn.ReLU(),
        )
        # Use NoisyLinear even if only one layer uses it, as per convention
        self.value_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.num_atoms) # Output V(s) distribution logits
        )
        self.advantage_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.action_dim * self.num_atoms) # Output A(s,a) distribution logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.feature_layer(x)

        values = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantages = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)

        # Combine value and advantage streams (Dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_logits = values + advantages - advantages.mean(dim=1, keepdim=True)

        # Apply Softmax to get probability distributions for each action
        q_probs = F.softmax(q_logits, dim=2) # Softmax over atoms dimension

        # Avoid numerical instability (ensure probabilities sum to 1 and are > 0)
        q_probs = (q_probs + 1e-8) / (1.0 + self.num_atoms * 1e-8) # Normalize gently

        return q_probs # Return distributions [batch, action, atoms]

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


#####################
#    DQN Agent      #
#####################
class DQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms) # Distribution support (atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1) # Distance between atoms

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.support = self.support.to(self.device) # Move support to device

        # Networks (Now with C51/Noisy capabilities)
        self.policy_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.initial_learning_rate, eps=1e-5) # AdamW might be slightly better with NoisyNets

        # Replay Buffer (Prioritized)
        self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size, config.per_alpha, config) if config.use_per else None
        if not config.use_per:
             print("Warning: PER disabled, using standard Replay Buffer.")
             # Ensure standard buffer uses the correct Experience tuple if PER is off
             self.replay_buffer = deque([], maxlen=config.replay_buffer_size) # Simple deque for N-step tuples

        # Normalization helpers are now in SumoEnv

        # Training step counter (for target net updates)
        self.train_step_count = 0
        self.loss_history = [] # Track loss for logging

    def get_action(self, normalized_state: np.ndarray, current_lane_idx: int) -> int:
        """Choose action based on expected Q-values from the (noisy) policy network"""
        # Reset noise before action selection if using Noisy Nets
        if self.config.use_noisy_nets:
            self.policy_net.reset_noise()
            # No need to reset target_net noise here as it's not used for action selection

        self.policy_net.eval() # Set to evaluation mode for inference (important for Noisy Nets eval)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            # Get action probability distributions [1, action_dim, num_atoms]
            action_probs = self.policy_net(state_tensor)
            # Calculate expected Q-values: sum(probability * support_value) for each action
            expected_q_values = (action_probs * self.support).sum(dim=2) # [1, action_dim]

            # Action Masking: Invalidate Q-values for impossible actions
            num_lanes = len(self.config.lane_max_speeds)
            q_values_masked = expected_q_values.clone()
            if current_lane_idx == 0:
                q_values_masked[0, 1] = -float('inf') # Cannot turn left
            if current_lane_idx >= num_lanes - 1:
                q_values_masked[0, 2] = -float('inf') # Cannot turn right

            action = q_values_masked.argmax().item() # Choose action with highest valid expected Q-value

        self.policy_net.train() # Set back to training mode
        return action

    def update(self, global_step: int) -> Optional[float]:
        """Update the Q-network using a batch from the replay buffer (PER + N-Step + C51 + Noisy aware + Fixed DoubleDQN masking)"""
        if len(self.replay_buffer) < self.config.learning_starts: return None
        if len(self.replay_buffer) < self.config.batch_size: return None

        current_beta = linear_decay(self.config.per_beta_start, self.config.per_beta_end,
                                    self.config.per_beta_annealing_steps, global_step)

        # Sample from buffer
        try:
            if self.config.use_per:
                experiences, is_weights, indices = self.replay_buffer.sample(self.config.batch_size, current_beta)
            else: # Sample uniformly if PER is off
                experiences = random.sample(self.replay_buffer, self.config.batch_size)
                is_weights = np.ones(self.config.batch_size) # No IS weights needed
                indices = None # No indices needed
        except RuntimeError as e:
             print(f"Error sampling from replay buffer: {e}. Skipping update.")
             return None


        batch = Experience(*zip(*experiences))

        states_np = np.array(batch.state)
        actions_np = np.array(batch.action)
        n_step_rewards_np = np.array(batch.n_step_reward) # Rewards are already normalized by env.step
        next_states_np = np.array(batch.next_state)
        dones_np = np.array(batch.done)
        n_np = np.array(batch.n)
        next_lane_indices_np = np.array(batch.next_lane_index) # *** Get next lane indices ***

        # Convert to tensors
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).to(self.device) # [batch_size]
        rewards = torch.FloatTensor(n_step_rewards_np).to(self.device) # [batch_size]
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.BoolTensor(dones_np).to(self.device) # [batch_size]
        is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)
        gammas = torch.FloatTensor(self.config.gamma ** n_np).to(self.device) # [batch_size]
        next_lane_indices = torch.LongTensor(next_lane_indices_np).to(self.device) # [batch_size]

        # --- Calculate Target Distribution (C51 logic) ---
        with torch.no_grad():
            # Get next state distributions from target network
            if self.config.use_noisy_nets: self.target_net.reset_noise()
            next_dist_target = self.target_net(next_states) # [batch, action, atoms]

            # --- Double DQN selection ---
            if self.config.use_double_dqn:
                # Use policy network to select best actions for next state
                if self.config.use_noisy_nets: self.policy_net.reset_noise()
                next_dist_policy = self.policy_net(next_states) # [batch, action, atoms]
                next_expected_q_policy = (next_dist_policy * self.support).sum(dim=2) # [batch, action]

                # *** Mask Q-values for invalid actions in the *next* state ***
                num_lanes = len(self.config.lane_max_speeds)
                q_values_masked = next_expected_q_policy.clone()
                # Mask left turn if in leftmost lane (index 0)
                q_values_masked[next_lane_indices == 0, 1] = -float('inf')
                # Mask right turn if in rightmost lane (index num_lanes - 1)
                q_values_masked[next_lane_indices >= num_lanes - 1, 2] = -float('inf')

                # Select best valid actions according to policy net Q-values
                best_next_actions = q_values_masked.argmax(dim=1) # [batch_size]

            else:
                 # Standard DQN selection (using target net)
                 next_expected_q_target = (next_dist_target * self.support).sum(dim=2) # [batch, action]
                 # Apply masking here as well
                 num_lanes = len(self.config.lane_max_speeds)
                 q_values_masked = next_expected_q_target.clone()
                 q_values_masked[next_lane_indices == 0, 1] = -float('inf')
                 q_values_masked[next_lane_indices >= num_lanes - 1, 2] = -float('inf')
                 best_next_actions = q_values_masked.argmax(dim=1) # [batch_size]

            # Get the target network's distribution corresponding to the chosen best next actions
            best_next_dist = next_dist_target[range(self.config.batch_size), best_next_actions, :] # [batch, atoms]

            # --- Project Target Distribution ---
            # Compute projected support: Tz = R + gamma^N * z
            Tz = rewards.unsqueeze(1) + gammas.unsqueeze(1) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            # Compute indices and offsets for projection
            b = (Tz - self.v_min) / self.delta_z
            lower_idx = b.floor().long()
            upper_idx = b.ceil().long()

            # Ensure indices are valid after potential floating point issues at boundaries
            lower_idx = lower_idx.clamp(min=0, max=self.num_atoms - 1)
            upper_idx = upper_idx.clamp(min=0, max=self.num_atoms - 1)

            # Distribute probabilities using index_add_ for efficiency
            target_dist_projected = torch.zeros_like(best_next_dist) # [batch, atoms]
            # Calculate weights for lower and upper indices
            # Weight for lower index = (upper_idx - b)
            # Weight for upper index = (b - lower_idx)
            lower_weight = (upper_idx.float() - b).clamp(min=0, max=1) # Proba contribution to lower bin
            upper_weight = (b - lower_idx.float()).clamp(min=0, max=1) # Proba contribution to upper bin

            # Use index_add_ for projection (more efficient than loops)
            # Reshape for index_add: needs flat indices and flat values
            batch_indices = torch.arange(self.config.batch_size, device=self.device).unsqueeze(1).expand_as(lower_idx)

            # Flatten tensors for index_add_ along dimension 1 (atoms)
            flat_lower_idx = lower_idx.flatten()
            flat_upper_idx = upper_idx.flatten()
            flat_best_next_dist = best_next_dist.flatten()
            flat_lower_weight = lower_weight.flatten()
            flat_upper_weight = upper_weight.flatten()
            flat_batch_indices = batch_indices.flatten()

            # Project onto lower indices
            target_dist_projected.index_put_(
                (flat_batch_indices, flat_lower_idx),
                flat_best_next_dist * flat_lower_weight,
                accumulate=True
            )
            # Project onto upper indices
            target_dist_projected.index_put_(
                (flat_batch_indices, flat_upper_idx),
                flat_best_next_dist * flat_upper_weight,
                accumulate=True
            )


        # --- Calculate Current Distribution for Chosen Actions ---
        if self.config.use_noisy_nets: self.policy_net.reset_noise()
        current_dist_all_actions = self.policy_net(states) # [batch, action, atoms]
        current_dist = current_dist_all_actions[range(self.config.batch_size), actions, :] # [batch, atoms]

        # --- Calculate Loss (Cross-Entropy) ---
        # Ensure current_dist has no zeros before log
        elementwise_loss = -(target_dist_projected.detach() * (current_dist + 1e-7).log()).sum(dim=1) # [batch]

        # --- Calculate Priorities for PER ---
        # Use absolute TD error (elementwise loss) for priority
        td_errors = elementwise_loss.detach()

        # Update priorities in the buffer if using PER
        if self.config.use_per and indices is not None:
             self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        # --- Apply IS Weights and Calculate Final Loss ---
        loss = (elementwise_loss * is_weights_tensor).mean()

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping helps stabilize C51/NoisyNets
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.loss_history.append(loss.item())

        # --- Update Target Network ---
        if self.train_step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"---- Target Network Updated (Step: {self.train_step_count}) ----") # Add confirmation log


        return loss.item()

#####################
#   Main Training Loop #
#####################
def main():
    config = Config()

    # --- Basic Checks & Setup ---
    if not os.path.exists(config.config_path): print(f"Warning: {config.config_path} not found.")

    # --- Environment and Agent Initialization ---
    env = SumoEnv(config)      # SUMO environment (handles normalization internally)
    agent = DQNAgent(config)   # DQN Agent (C51/Noisy/PER/N-Step enabled)

    # --- Logging and Saving Setup ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = "dqn"
    if config.use_per: exp_name += "_per"
    if config.use_n_step: exp_name += f"_n{config.n_step}"
    if config.use_distributional: exp_name += "_c51"
    if config.use_noisy_nets: exp_name += "_noisy"
    exp_name += "_revised" # Indicate revised version
    results_dir = f"{exp_name}_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")
    print(f"Config: PER={config.use_per}, N-Step={config.n_step if config.use_n_step else 'Off'}, C51={config.use_distributional}, Noisy={config.use_noisy_nets}")
    print(f"LR={config.initial_learning_rate}, TargetUpdate={config.target_update_freq}, StartLearn={config.learning_starts}")
    print(f"Reward Params: FastLaneBonus={config.reward_faster_lane_bonus}, StaySlowPen={config.reward_staying_slow_penalty_scale}, LC_Penalty={config.reward_lane_change_penalty}, SpeedScale={config.reward_high_speed_scale}")


    # Save config used for this run
    try:
        config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        config_save_path = os.path.join(results_dir, "config_used.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
             json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
        print(f"Configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"Warning: Could not save config JSON: {e}")

    # Metrics Lists
    all_rewards_sum_norm = [] # Track SUM of NORMALIZED rewards per episode
    lane_change_counts = []
    collision_counts = []
    total_steps_per_episode = []
    avg_losses_per_episode = []
    # epsilon_values = [] # No longer used with Noisy Nets
    beta_values = [] # Track PER beta
    best_avg_reward = -float('inf')
    global_step_count = 0

    # N-Step Buffer
    n_step_buffer = deque(maxlen=config.n_step)

    # --- DQN Training Loop ---
    print("\n" + "#"*20 + f" Starting DQN Training ({exp_name}) " + "#"*20)
    try:
        for episode in tqdm(range(1, config.dqn_episodes + 1), desc="DQN Training Episodes"):
            state_norm = env.reset() # Get initial NORMALIZED state
            # Ensure state_norm is valid before starting episode
            if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                print(f"ERROR: Initial state is invalid in episode {episode}. Resetting again...")
                time.sleep(1)
                state_norm = env.reset()
                if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                    print("FATAL: Still invalid state after reset. Aborting.")
                    raise RuntimeError("Invalid initial state.")

            episode_reward_norm_sum = 0.0 # Sum of NORMALIZED rewards this episode
            episode_loss_sum = 0.0
            episode_loss_count = 0
            done = False
            step_count = 0
            n_step_buffer.clear() # Clear n-step buffer for new episode

            while not done and step_count < config.max_steps:
                # Get current lane index from the *raw* state for accurate masking
                current_lane_idx = int(round(env.last_raw_state[1]))
                num_lanes = len(config.lane_max_speeds)
                # Clip index just in case raw state was momentarily invalid
                current_lane_idx = np.clip(current_lane_idx, 0, num_lanes - 1)

                # Get action based on normalized state and current lane
                action = agent.get_action(state_norm, current_lane_idx)

                # Step environment -> returns NORMALIZED next state, NORMALIZED reward, done, NEXT lane index
                next_state_norm, norm_reward, done, next_lane_idx = env.step(action)

                 # Validate data from env.step before adding to buffer
                if np.any(np.isnan(state_norm)) or np.any(np.isnan(next_state_norm)) or not np.isfinite(norm_reward):
                     print(f"Warning: Invalid data detected from env.step at Ep {episode}, Step {step_count}. Skipping buffer push.")
                     # Decide how to handle: skip step, end episode, use last valid state?
                     # Let's skip pushing this transition and continue with last valid state
                     # Note: state_norm might be invalid here too if step failed badly
                     if np.any(np.isnan(next_state_norm)): next_state_norm = state_norm # Keep old state if next is bad
                     done = True # End episode if state becomes invalid
                     # Need to decide if we still process N-step buffer here...
                else:
                    # Store transition in N-step buffer (using normalized states/rewards and next lane index)
                    n_step_buffer.append(NStepTransition(state_norm, action, norm_reward, next_state_norm, done, next_lane_idx))

                    # --- N-Step Experience Generation ---
                    # Generate experience if buffer is full OR episode is done (and buffer has items)
                    if len(n_step_buffer) >= config.n_step or (done and len(n_step_buffer) > 0):
                        n_step_actual = len(n_step_buffer) # Actual number of steps in this transition
                        n_step_return_discounted = 0.0
                        # Calculate N-step return using normalized rewards stored in buffer
                        for i in range(n_step_actual):
                            # gamma^i * r_{t+i} (where r is normalized reward)
                            n_step_return_discounted += (config.gamma**i) * n_step_buffer[i].reward

                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        # Final next state is s_{t+n} from the last transition in the buffer
                        s_t_plus_n = n_step_buffer[-1].next_state
                        # Terminal state reached if the last transition in buffer ended the episode
                        done_n_step = n_step_buffer[-1].done
                        # Get the lane index corresponding to s_t_plus_n
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index

                        exp = Experience(state=s_t, action=a_t, n_step_reward=n_step_return_discounted,
                                         next_state=s_t_plus_n, done=done_n_step, n=n_step_actual,
                                         next_lane_index=next_lane_idx_n_step) # Add next_lane_idx

                        # Add experience to replay buffer
                        try:
                           if config.use_per: agent.replay_buffer.push(exp)
                           else: agent.replay_buffer.append(exp)
                        except Exception as e_push:
                             print(f"Error pushing experience to buffer: {e_push}")
                             traceback.print_exc()
                             # Decide if we should continue or stop

                        # If N-step experience was generated because buffer was full (not 'done'), remove oldest item
                        if len(n_step_buffer) >= config.n_step:
                            n_step_buffer.popleft()

                        # If episode ended ('done' is True), the buffer will be cleared later outside the while loop or flushed below.


                # Update state only if the step was valid
                if not (np.any(np.isnan(next_state_norm)) or np.any(np.isinf(next_state_norm))):
                    state_norm = next_state_norm
                else:
                    # Handle invalid next state - maybe keep old state or terminate
                    print(f"Warning: Keeping previous state due to invalid next_state_norm at Ep {episode}, Step {step_count}")
                    # done = True # Optionally terminate

                # Accumulate NORMALIZED reward for logging
                episode_reward_norm_sum += norm_reward

                step_count += 1
                global_step_count += 1

                # Perform DQN update
                loss = agent.update(global_step_count)
                if loss is not None:
                    episode_loss_sum += loss
                    episode_loss_count += 1

                # Check collision flag explicitly (redundant if done is set, but safe)
                if env.collision_occurred: done = True

                # --- Process remaining N-step transitions if episode ended ---
                # This needs refinement: If done, we need to generate experiences for all remaining items in buffer
                if done and len(n_step_buffer) > 0:
                     # Keep generating experiences until the buffer is empty
                     while len(n_step_buffer) > 0:
                        n_step_actual = len(n_step_buffer)
                        n_step_return_discounted = 0.0
                        for i in range(n_step_actual):
                            n_step_return_discounted += (config.gamma**i) * n_step_buffer[i].reward
                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        s_t_plus_n = n_step_buffer[-1].next_state
                        done_n_step = True # Episode ended
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index

                        exp = Experience(state=s_t, action=a_t, n_step_reward=n_step_return_discounted,
                                         next_state=s_t_plus_n, done=done_n_step, n=n_step_actual,
                                         next_lane_index=next_lane_idx_n_step)
                        try:
                           if config.use_per: agent.replay_buffer.push(exp)
                           else: agent.replay_buffer.append(exp)
                        except Exception as e_push:
                             print(f"Error pushing final experience to buffer: {e_push}")
                             traceback.print_exc()

                        n_step_buffer.popleft() # Remove processed transition


            # --- Episode End ---
            all_rewards_sum_norm.append(episode_reward_norm_sum) # Logging sum of normalized rewards
            lane_change_counts.append(env.change_lane_count)
            collision_counts.append(1 if env.collision_occurred else 0)
            total_steps_per_episode.append(step_count)
            avg_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0
            avg_losses_per_episode.append(avg_loss)

            # Get PER Beta for logging
            current_beta = linear_decay(config.per_beta_start, config.per_beta_end,
                                        config.per_beta_annealing_steps, global_step_count)
            beta_values.append(current_beta)


            # --- Save Best Model ---
            avg_window = 20 # Rolling average window
            if episode >= avg_window:
                # Use the summed normalized rewards for tracking best model
                rewards_slice = all_rewards_sum_norm[max(0, episode - avg_window):episode]
                if rewards_slice:
                   current_avg_reward = np.mean(rewards_slice)
                   if current_avg_reward > best_avg_reward:
                       best_avg_reward = current_avg_reward
                       best_model_path = os.path.join(models_dir, "best_model.pth")
                       torch.save(agent.policy_net.state_dict(), best_model_path)
                       print(f"\n🎉 New best avg reward ({avg_window}ep, summed norm): {best_avg_reward:.2f}! Model saved.")
                else: current_avg_reward = -float('inf')


            # --- Periodic Logging ---
            if episode % config.log_interval == 0:
                 log_slice_start = max(0, episode - config.log_interval)
                 avg_reward_log = np.mean(all_rewards_sum_norm[log_slice_start:]) if all_rewards_sum_norm[log_slice_start:] else 0
                 avg_steps_log = np.mean(total_steps_per_episode[log_slice_start:]) if total_steps_per_episode[log_slice_start:] else 0
                 collision_rate_log = np.mean(collision_counts[log_slice_start:]) * 100 if collision_counts[log_slice_start:] else 0
                 avg_loss_log = np.mean(avg_losses_per_episode[log_slice_start:]) if avg_losses_per_episode[log_slice_start:] else 0
                 avg_lc_log = np.mean(lane_change_counts[log_slice_start:]) if lane_change_counts[log_slice_start:] else 0 # Log lane changes

                 print(f"\nEp: {episode}/{config.dqn_episodes} | Avg Reward (last {config.log_interval}, summed norm): {avg_reward_log:.2f} "
                       f"| Best Avg Reward ({avg_window}ep): {best_avg_reward:.2f} "
                       f"| Avg Steps: {avg_steps_log:.1f} | Avg LC: {avg_lc_log:.1f} " # Added LC log
                       f"| Coll Rate: {collision_rate_log:.1f}% "
                       f"| Avg Loss: {avg_loss_log:.4f} | Beta: {current_beta:.3f} | Steps: {global_step_count}")

            # --- Periodic Model Saving ---
            if episode % config.save_interval == 0:
                 periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                 torch.save(agent.policy_net.state_dict(), periodic_model_path)

    except KeyboardInterrupt: print("\nTraining interrupted by user.")
    except Exception as e: print(f"\nFATAL ERROR during training: {e}"); traceback.print_exc()
    finally:
        print("Closing final environment...")
        env._close()

        # --- Save Final Model and Data ---
        if 'agent' in locals() and agent is not None:
            last_model_path = os.path.join(models_dir, "last_model.pth")
            torch.save(agent.policy_net.state_dict(), last_model_path)
            print(f"Final model saved to: {last_model_path}")

            # --- Plotting (Keeping original structure) ---
            print("Generating training plots...")
            plt.figure("DQN Training Curves", figsize=(15, 22)) # Same figure size

            # Plot 1: Reward (Summed Normalized) + Rolling Mean
            plt.subplot(5, 2, 1); plt.grid(True, linestyle='--')
            plt.plot(all_rewards_sum_norm, label='Episode Reward (Summed Norm)', alpha=0.6)
            if len(all_rewards_sum_norm) >= 10:
                rolling_avg = np.convolve(all_rewards_sum_norm, np.ones(10)/10, mode='valid')
                plt.plot(np.arange(9, len(all_rewards_sum_norm)), rolling_avg, label='10-Ep Roll Avg', color='red', linewidth=2)
            plt.title("Episode Reward (Summed Normalized)"); plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()

            # Plot 2: Lane Changes
            plt.subplot(5, 2, 2); plt.grid(True, linestyle='--')
            plt.plot(lane_change_counts); plt.title("Lane Changes per Episode")
            plt.xlabel("Episode"); plt.ylabel("Count")

            # Plot 3: Collisions + Rolling Rate
            plt.subplot(5, 2, 3); plt.grid(True, linestyle='--')
            collision_counts_np = np.array(collision_counts)
            episodes_axis = np.arange(1, len(collision_counts_np) + 1)
            collision_points = episodes_axis[collision_counts_np == 1]
            plt.plot(collision_points, np.ones_like(collision_points), marker='.', linestyle='None', alpha=0.6, label='Collision (1=Yes)', color='blue')
            if len(collision_counts) >= 10:
                 rolling_coll_rate = np.convolve(collision_counts, np.ones(10)/10, mode='valid') * 100
                 plt.plot(np.arange(10, len(collision_counts)+1), rolling_coll_rate, label='10-Ep Roll Coll Rate (%)', color='orange')
            plt.title("Collisions & Rolling Rate"); plt.xlabel("Episode"); plt.ylabel("Collision (0/1) / Rate (%)")
            plt.ylim(-5, 105); plt.legend()


            # Plot 4: Average DQN Loss per Episode (Cross-Entropy)
            plt.subplot(5, 2, 4); plt.grid(True, linestyle='--')
            valid_losses = [l for l in avg_losses_per_episode if l > 0]
            valid_indices = [i for i, l in enumerate(avg_losses_per_episode) if l > 0]
            if valid_indices:
                 plt.plot(valid_indices, valid_losses, label='Avg Loss per Episode', alpha=0.8)
                 if len(valid_losses) >= 10:
                     rolling_loss_avg = np.convolve(valid_losses, np.ones(10)/10, mode='valid')
                     plt.plot(valid_indices[9:], rolling_loss_avg, label='10-Ep Roll Avg Loss', color='purple', linewidth=1.5)
            plt.title("Average DQN Loss per Episode (Cross-Entropy)"); plt.xlabel("Episode"); plt.ylabel("Avg Loss"); plt.legend()
            # Consider log scale if loss varies widely: plt.yscale('log')

            # Plot 5: Steps per Episode
            plt.subplot(5, 2, 5); plt.grid(True, linestyle='--')
            plt.plot(total_steps_per_episode); plt.title("Steps per Episode")
            plt.xlabel("Episode"); plt.ylabel("Steps")

            # Plot 6: PER Beta Annealing
            plt.subplot(5, 2, 6); plt.grid(True, linestyle='--')
            plt.plot(beta_values); plt.title("PER Beta Annealing")
            plt.xlabel("Episode"); plt.ylabel("Beta Value")

            # Plot 7: Detailed Loss History (per training step)
            plt.subplot(5, 2, 7); plt.grid(True, linestyle='--')
            if agent.loss_history:
                steps_per_loss = np.arange(len(agent.loss_history))
                window_size = 100 # Rolling average over N training steps
                if len(agent.loss_history) >= window_size:
                    rolling_loss = np.convolve(agent.loss_history, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(steps_per_loss[window_size-1:], rolling_loss, label=f'{window_size}-Step Roll Avg Loss')
                else:
                    plt.plot(steps_per_loss, agent.loss_history, label='Loss per training step', alpha=0.5)
                plt.title("DQN Loss per Training Step (Cross-Entropy)"); plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.legend()
            else: plt.title("DQN Loss per Training Step (No data)")

            # Plot 8: (Reserved Plot Slot)
            plt.subplot(5, 2, 8); plt.grid(True, linestyle='--')
            plt.title("Reserved Plot Slot 8")

            # Plot 9 & 10: Free slots (unchanged)
            # plt.subplot(5, 2, 9) ...
            # plt.subplot(5, 2, 10) ...


            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout
            plot_path = os.path.join(results_dir, "dqn_training_curves_final_revised.png")
            plt.savefig(plot_path)
            plt.close("all")
            print(f"Training plots saved to: {plot_path}")

            # --- Save Training Data JSON ---
            print("Saving training data...")
            training_data = {
                "episode_rewards_sum_norm": all_rewards_sum_norm, # Name reflects it's summed normalized rewards
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "avg_losses_per_episode": avg_losses_per_episode,
                "beta_values_per_episode": beta_values,
                "detailed_loss_history_per_step": agent.loss_history
            }
            data_path = os.path.join(results_dir, "training_data_final_revised.json")
            try:
                # Use a robust serializer for numpy types
                def default_serializer(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, np.bool_): return bool(obj)
                    elif isinstance(obj, (datetime.datetime, datetime.date)): return obj.isoformat()
                    try: return json.JSONEncoder.default(self, obj) # Fallback
                    except TypeError: return str(obj) # Last resort: convert to string


                # Clean data before saving (ensure basic types)
                for key in training_data:
                     if isinstance(training_data[key], list) and len(training_data[key]) > 0:
                         # Convert numpy types within lists
                         training_data[key] = [default_serializer(item) for item in training_data[key]]
                         # Handle potential None or NaN in losses
                         if key == "avg_losses_per_episode" or key == "detailed_loss_history_per_step":
                              training_data[key] = [0.0 if (item is None or not np.isfinite(item)) else float(item) for item in training_data[key]]


                with open(data_path, "w", encoding="utf-8") as f:
                     json.dump(training_data, f, indent=4, ensure_ascii=False, default=default_serializer)
                print(f"Training data saved to: {data_path}")
            except Exception as e:
                print(f"ERROR saving training data: {e}")
                traceback.print_exc()
        else:
            print("Agent not initialized, cannot save final model/data.")

        print(f"\n DQN Training ({exp_name}) finished. Results saved in: {results_dir}")

if __name__ == "__main__":
    if not os.path.exists("a.sumocfg"): print("Warning: a.sumocfg not found.")
    if not os.path.exists("a.net.xml"): print("Warning: a.net.xml not found.")
    if not os.path.exists("a.rou.xml"): print("Warning: a.rou.xml not found.")
    main()