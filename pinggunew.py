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
from collections import deque, namedtuple, Counter
from typing import List, Tuple, Dict, Optional, Any
import socket
import traceback
import collections
import math
import copy # For deep copying configs/normalizers

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

# --- 导入必要的组件 ---
try:
    # Import necessary classes from training scripts
    from dqn import Config as DQN_Config_Base
    from dqn import QNetwork, NoisyLinear, RunningMeanStd as DQN_RunningMeanStd
    from ppo import Config as PPO_Config_Base
    from ppo import PPO, BehaviorCloningNet # PPO needs BC net only if loading weights from BC init
    from ppo import RunningMeanStd as PPO_RunningMeanStd

except ImportError as e:
    print(f"Error importing from dqn.py or ppo.py: {e}")
    print("Please ensure dqn.py and ppo.py are in the same directory or accessible in the Python path.")
    sys.exit(1)
except AttributeError as e_attr:
     print(f"AttributeError during import (maybe a class name changed?): {e_attr}")
     sys.exit(1)


#####################################
# 评估配置                         #
#####################################
class EvalConfig:
    # --- 模型路径 ---
    DQN_MODEL_PATH = "dqn.pth" # 修改为您的 DQN 模型路径
    PPO_MODEL_PATH = "ppo.pth" # 修改为您的 PPO 模型路径

    # --- SUMO 配置 ---
    # --->>> 确保这里使用 sumo-gui 以便观察颜色 <<<---
    EVAL_SUMO_BINARY = "sumo-gui"  # Use GUI for visualization: "sumo-gui"
    EVAL_SUMO_CONFIG = "new.sumocfg" # Use the new config file
    EVAL_STEP_LENGTH = 0.2         # Should match training step length
    EVAL_PORT_RANGE = (8910, 8920) # Different port range from training

    # --- 评估参数 ---
    EVAL_EPISODES = 5              # Number of episodes per model (adjust as needed, start small for testing)
    EVAL_MAX_STEPS = 1500          # Max steps per evaluation episode (e.g., 300 seconds)
    EVAL_SEED = 42                 # Seed for SumoEnv startup if needed for reproducibility
    NUM_LANES = 4                  # Number of lanes in new.net.xml (0, 1, 2, 3)
    EGO_INSERTION_DELAY_SECONDS = 5.0 # Shorter delay for faster eval start (adjust if needed)

    # --- 强制换道尝试逻辑 ---
    # EVALUATION MODE: ONLY CHANGE LANE WHEN FORCED AND AGREED
    FORCE_CHANGE_INTERVAL_STEPS = 50 # Propose a forced change every X steps (e.g., 10 seconds at 0.2 step)
    FORCE_CHANGE_MONITOR_STEPS = 15  # Steps to wait/monitor completion of *agreed* change (3 seconds)
    FORCE_CHANGE_SUCCESS_DIST = 1.0 # Min lateral distance to consider change successful (adjust if needed)

    # --- 归一化 ---
    # Normalizer state should ideally be loaded alongside the models.
    # For this script, we assume the train configs correctly state if norm was used.
    # We will re-create the normalizer objects but NOT update them during eval.

    # --- Ego Vehicle ID & Type for Evaluation ---
    EVAL_EGO_ID = "eval_ego"
    EVAL_EGO_TYPE = "car_ego_eval" # Use a distinct type ID for eval

#####################################
# 辅助函数 (复制/改编)              #
#####################################
def get_available_port(start_port, end_port):
    """Find an available port in the specified range"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError(f"No available port found in range [{start_port}, {end_port}].")

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
    except Exception as e: print(f"Warning: Error killing SUMO processes: {e}")
    time.sleep(0.1)

# Use RunningMeanStd from one of the training scripts (they should be identical)
RunningMeanStd = DQN_RunningMeanStd

# --- Helper for Plotting Rolling Average ---
def calculate_rolling_average(data, window):
    if len(data) < window:
        return np.array([]) # Not enough data for a full window
    data_np = np.array(data, dtype=float)
    if np.any(np.isnan(data_np)):
         print(f"Warning: NaNs detected in rolling average input data. Result may be inaccurate.")
         data_np = np.nan_to_num(data_np, nan=np.nanmean(data_np)) # Replace NaN with mean

    if len(data_np) < window: # Check again after potential NaN removal
         return np.array([])

    return np.convolve(data_np, np.ones(window) / window, mode='valid')


#####################################
# 评估环境封装                       #
#####################################
class EvaluationEnv:
    # --- NO CHANGES NEEDED in EvaluationEnv ---
    # The environment just executes the action it receives.
    # The logic for *deciding* the action is in evaluate_episode.
    def __init__(self, eval_config: EvalConfig, dqn_train_config: DQN_Config_Base, sumo_seed: int):
        self.eval_config = eval_config
        self.dqn_train_config = dqn_train_config # Need this for state dim, max distance etc.
        self.sumo_binary = self.eval_config.EVAL_SUMO_BINARY
        self.config_path = self.eval_config.EVAL_SUMO_CONFIG
        self.step_length = self.eval_config.EVAL_STEP_LENGTH
        self.ego_vehicle_id = self.eval_config.EVAL_EGO_ID
        self.ego_type_id = self.eval_config.EVAL_EGO_TYPE
        self.port_range = self.eval_config.EVAL_PORT_RANGE
        self.num_lanes = self.eval_config.NUM_LANES
        self.sumo_seed = sumo_seed
        self.ego_insertion_delay_steps = int(self.eval_config.EGO_INSERTION_DELAY_SECONDS / self.step_length)

        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_raw_state = np.zeros(self.dqn_train_config.state_dim)
        self.current_step = 0
        self.collision_occurred = False
        self.ego_start_pos = None
        self.ego_route_id = "route_E0" # Assuming the route from new.rou.xml
        self.last_ego_pos = None # Store last known position

    def _start_sumo(self):
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.port_range[0], self.port_range[1])
        except IOError as e:
             print(f"Error: Could not find available port: {e}"); sys.exit(1)

        sumo_cmd = [
            self.sumo_binary, "-c", self.config_path,
            "--remote-port", str(self.traci_port), "--step-length", str(self.step_length),
            "--collision.check-junctions", "true", "--collision.action", "warn", # Warn allows detection
            "--time-to-teleport", "-1", "--no-warnings", "true", "--seed", str(self.sumo_seed)
        ]
        if self.sumo_binary == "sumo-gui":
            sumo_cmd.extend(["--quit-on-end", "true"])

        try:
             stdout_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             # print(f"Starting SUMO with command: {' '.join(sumo_cmd)}") # Debug print
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
        except FileNotFoundError: print(f"Error: SUMO executable '{self.sumo_binary}' not found."); sys.exit(1)
        except Exception as e: print(f"Error: Failed to start SUMO process: {e}"); sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5) # Increased sleep
                traci.init(self.traci_port)
                # print(f"SUMO TraCI connected on port {self.traci_port}.") # Debug print
                return
            except traci.exceptions.TraCIException as te:
                # print(f"TraCI connection attempt {attempt+1} failed: {te}") # Debug print
                if attempt == connection_attempts - 1:
                    print("Max TraCI connection attempts reached."); self._close(); raise ConnectionError(f"Could not connect to SUMO (Port: {self.traci_port}).")
            except Exception as e: print(f"Unexpected error connecting TraCI: {e}"); self._close(); raise ConnectionError(f"Unknown error connecting to SUMO (Port: {self.traci_port}).")

    # --->>> START OF MODIFIED _add_ego_vehicle <<<---
    def _add_ego_vehicle(self):
        """Add the ego vehicle to the simulation with evaluation-specific type/color."""
        # Ensure route exists
        if self.ego_route_id not in traci.route.getIDList():
            edge_list = list(traci.edge.getIDList())
            first_edge = edge_list[0] if edge_list else None
            if not first_edge or first_edge != "E0":
                first_edge = "E0" if "E0" in edge_list else first_edge
            if first_edge:
                print(f"Warning: Route '{self.ego_route_id}' not found. Creating route from edge '{first_edge}'.")
                try:
                    traci.route.add(self.ego_route_id, [first_edge])
                except traci.exceptions.TraCIException as e_add:
                    raise RuntimeError(f"Failed to add route '{self.ego_route_id}' using edge '{first_edge}': {e_add}")
            else:
                raise RuntimeError(
                    f"Route '{self.ego_route_id}' not found and no suitable edge available to create it.")

        # Ensure vehicle type exists or create it
        type_created_or_exists = False # Flag to track if type is ready
        if self.ego_type_id not in traci.vehicletype.getIDList():
            base_type = "car"
            if base_type not in traci.vehicletype.getIDList():
                available_types = traci.vehicletype.getIDList()
                base_type = available_types[0] if available_types else None
                if not base_type: raise RuntimeError("No vehicle types found in simulation to base ego type on.")
                print(f"Warning: Base type 'car' not found. Using fallback '{base_type}'.")

            try:
                print(f"Attempting to create Ego type '{self.ego_type_id}' by copying from '{base_type}'...")
                traci.vehicletype.copy(base_type, self.ego_type_id)
                print(f"Type '{self.ego_type_id}' copied. Now setting parameters...")
                # --->>> 修改点 1: 使用整数 RGB 颜色格式 <<<---
                color_str = "0,0,204" # 深蓝色 (R, G, B integers 0-255)
                traci.vehicletype.setParameter(self.ego_type_id, "color", color_str)
                # --->>> 修改点 2: 添加成功提示 <<<---
                print(f"Successfully attempted to set 'color={color_str}' parameter for type '{self.ego_type_id}'.")

                traci.vehicletype.setParameter(self.ego_type_id, "lcStrategic", "1.0")
                traci.vehicletype.setParameter(self.ego_type_id, "lcSpeedGain", "1.0")
                traci.vehicletype.setParameter(self.ego_type_id, "lcCooperative", "0.5")
                print(f"Set other parameters for Ego type '{self.ego_type_id}'.")
                type_created_or_exists = True # Mark as created successfully
            except traci.exceptions.TraCIException as e:
                # --->>> 修改点 3: 打印更详细的错误信息 <<<---
                print(f"ERROR: Failed to set parameters (including color) for Ego type '{self.ego_type_id}': {e}")
                # Even if setting params failed, the type might exist, proceed cautiously
                if self.ego_type_id in traci.vehicletype.getIDList():
                    type_created_or_exists = True # Type exists, though params might be default
            except Exception as e_generic:
                print(f"ERROR: Unexpected error during type creation/parameter setting for '{self.ego_type_id}': {e_generic}")

        else:
            print(f"Ego type '{self.ego_type_id}' already exists. Assuming parameters are set correctly.")
            type_created_or_exists = True # Type already existed

        # Remove lingering ego vehicle if it exists
        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.ego_vehicle_id)
                time.sleep(0.1)
            except traci.exceptions.TraCIException as e:
                print(f"Warning: Failed to remove lingering Ego '{self.ego_vehicle_id}': {e}")

        # Add the evaluation ego vehicle only if the type is confirmed to exist
        if not type_created_or_exists:
             raise RuntimeError(f"Cannot add ego vehicle because type '{self.ego_type_id}' could not be created or confirmed.")

        try:
            start_lane = random.choice(range(self.num_lanes))
            print(f"Attempting to add ego '{self.ego_vehicle_id}' with type '{self.ego_type_id}' on route '{self.ego_route_id}', lane {start_lane}...")
            traci.vehicle.add(vehID=self.ego_vehicle_id, routeID=self.ego_route_id, typeID=self.ego_type_id,
                              depart="now", departLane=start_lane, departSpeed="max")

            # Wait for it to appear
            wait_steps = int(2.0 / self.step_length)
            ego_appeared = False
            for step in range(wait_steps):
                traci.simulationStep()
                if self.ego_vehicle_id in traci.vehicle.getIDList():
                    ego_appeared = True
                    self.ego_start_pos = traci.vehicle.getPosition(self.ego_vehicle_id)
                    self.last_ego_pos = self.ego_start_pos
                    print(f"Ego vehicle '{self.ego_vehicle_id}' appeared at step {step+1} after add.")

                    # --->>> 修改点 4: 添加直接设置车辆颜色的后备方法 <<<---
                    try:
                        # 使用 RGBA 元组 (0-255)
                        blue_color_tuple = (0, 0, 204, 255) # R, G, B, Alpha (255 = 不透明)
                        traci.vehicle.setColor(self.ego_vehicle_id, blue_color_tuple)
                        print(f"DEBUG: Explicitly set color for vehicle '{self.ego_vehicle_id}' using traci.vehicle.setColor.")
                    except traci.exceptions.TraCIException as e_setcolor:
                        print(f"DEBUG: Failed to explicitly set color for vehicle '{self.ego_vehicle_id}': {e_setcolor}")
                    except Exception as e_generic_setcolor:
                        print(f"DEBUG: Unexpected error setting color for vehicle '{self.ego_vehicle_id}': {e_generic_setcolor}")

                    # 检查实际类型 (可选调试)
                    try:
                        actual_type = traci.vehicle.getTypeID(self.ego_vehicle_id)
                        print(f"DEBUG: Ego vehicle '{self.ego_vehicle_id}' actual type ID is: '{actual_type}'. Expected: '{self.ego_type_id}'")
                    except traci.exceptions.TraCIException as e_gettype:
                         print(f"DEBUG: Could not get type ID for '{self.ego_vehicle_id}': {e_gettype}")

                    break # Exit loop once appeared
            if not ego_appeared:
                current_vehicles = []
                try: current_vehicles = traci.vehicle.getIDList()
                except: pass
                print(f"Current vehicles after wait: {current_vehicles}")
                raise RuntimeError(
                    f"Ego vehicle '{self.ego_vehicle_id}' did not appear within {wait_steps} steps after add (depart='now').")
        except traci.exceptions.TraCIException as e:
            print(f"Error: Failed adding Ego vehicle '{self.ego_vehicle_id}': {e}")
            try: print(f"Available routes: {traci.route.getIDList()}")
            except: pass
            try: print(f"Available vTypes: {traci.vehicletype.getIDList()}")
            except: pass
            raise RuntimeError("Failed to add Ego vehicle.")
    # --->>> END OF MODIFIED _add_ego_vehicle <<<---


    def reset(self) -> np.ndarray:
        """Reset environment for a new evaluation episode"""
        self._close()
        self._start_sumo()

        # Initial steps for traffic build-up and ego delay
        print(f"Running initial {self.ego_insertion_delay_steps} steps for traffic buildup...")
        for _ in range(self.ego_insertion_delay_steps): # Delay loop
            try:
                traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                print(f"TraCI Error during delay loop: {e}")
                raise ConnectionError("SUMO connection closed during delay period.")

        print("Adding Ego vehicle...")
        self._add_ego_vehicle() # Call the modified method

        self.current_step = 0
        self.collision_occurred = False
        self.last_raw_state = np.zeros(self.dqn_train_config.state_dim)

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                # Allow a few steps for the vehicle state to stabilize after insertion
                for _ in range(int(1.0 / self.step_length)): traci.simulationStep()
                if self.ego_vehicle_id in traci.vehicle.getIDList(): # Check again
                    self.last_raw_state = self._get_raw_state().copy()
                    if np.any(np.isnan(self.last_raw_state)):
                         print("Warning: Initial raw state contains NaN after reset and stabilization.")
                         self.last_raw_state = np.nan_to_num(self.last_raw_state)
                else: print("Warning: Ego vehicle disappeared during stabilization steps in reset.")
            except traci.exceptions.TraCIException as e:
                print(f"Warning: TraCI exception during initial state fetch in reset: {e}. Using zeros.")
            except Exception as e_gen:
                print(f"Warning: Generic exception during initial state fetch in reset: {e_gen}. Using zeros.")
        else:
            print("Warning: Ego vehicle not found immediately after add/wait/stabilize in reset.")

        return self.last_raw_state

    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """Get distance and relative speed of nearest vehicles (copied from dqn.py, no changes needed)"""
        max_dist = self.dqn_train_config.max_distance
        infos = { 'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0), 'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0), 'right_back': (max_dist, 0.0) }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos
        try:
            ego_road = traci.vehicle.getRoadID(ego_id);
            if not ego_road: return infos # Ego might be teleporting or off-road briefly
        except traci.exceptions.TraCIException: return infos # Error getting ego road

        num_lanes_on_edge = self.num_lanes # Use configured number of lanes

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                if traci.vehicle.getRoadID(veh_id) != ego_road: continue
                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)

                if not (0 <= veh_lane < num_lanes_on_edge): continue # Skip vehicles on invalid lanes

                dx = veh_pos[0] - ego_pos[0] # Longitudinal distance difference
                dy = veh_pos[1] - ego_pos[1] # Lateral distance difference
                distance = math.hypot(dx, dy) # Euclidean distance
                longitudinal_distance = abs(dx)

                if longitudinal_distance >= max_dist: continue # Filter by longitudinal distance first

                rel_speed = ego_speed - veh_speed # Relative speed

                if veh_lane == ego_lane: # Same lane
                    # Only consider vehicles ahead in the same lane
                    if dx > 0 and longitudinal_distance < infos['front'][0]:
                        infos['front'] = (longitudinal_distance, rel_speed)
                elif veh_lane == ego_lane - 1: # Left lane
                    if dx > 0 and longitudinal_distance < infos['left_front'][0]: # Left Front
                        infos['left_front'] = (longitudinal_distance, rel_speed)
                    elif dx <= 0 and longitudinal_distance < infos['left_back'][0]: # Left Back
                        infos['left_back'] = (longitudinal_distance, rel_speed)
                elif veh_lane == ego_lane + 1: # Right lane
                     if dx > 0 and longitudinal_distance < infos['right_front'][0]: # Right Front
                         infos['right_front'] = (longitudinal_distance, rel_speed)
                     elif dx <= 0 and longitudinal_distance < infos['right_back'][0]: # Right Back
                         infos['right_back'] = (longitudinal_distance, rel_speed)
            except traci.exceptions.TraCIException:
                # print(f"Warning: TraCI error getting info for vehicle {veh_id}. Skipping.") # Can be noisy
                continue
        return infos

    def _get_raw_state(self) -> np.ndarray:
        """Get current environment state (raw values before normalization, copied from dqn.py, no changes needed)"""
        state = np.zeros(self.dqn_train_config.state_dim, dtype=np.float32)
        ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            # print(f"Warning: Ego '{ego_id}' not in vehicle list during _get_raw_state. Returning last state.") # Debug
            return self.last_raw_state # Return last known raw state if ego disappeared

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road: # Could be between edges temporarily
                 # print(f"Warning: Ego '{ego_id}' has no current road ID. Returning last state.") # Debug
                 return self.last_raw_state

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            self.last_ego_pos = ego_pos # Update last known position

            # Clip lane index to be within valid range
            num_lanes = self.num_lanes
            ego_lane = np.clip(ego_lane, 0, num_lanes - 1)

            # Check lane change possibility *before* getting surrounding info, as it relies on current lane
            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # --- Populate Raw State Array ---
            state[0] = ego_speed
            state[1] = float(ego_lane) # Keep as float for consistency
            # Front vehicle info
            state[2] = min(surround_info['front'][0], self.dqn_train_config.max_distance)
            state[3] = surround_info['front'][1] # Relative speed
            # Left lane info
            state[4] = min(surround_info['left_front'][0], self.dqn_train_config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.dqn_train_config.max_distance)
            # Right lane info
            state[7] = min(surround_info['right_front'][0], self.dqn_train_config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.dqn_train_config.max_distance)
            # Lane change possibility flags
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"Warning: NaN or Inf detected in raw state calculation for {ego_id}. Using last valid raw state.")
                state = np.nan_to_num(state, nan=0.0, posinf=self.dqn_train_config.max_distance, neginf=-self.dqn_train_config.max_distance) # Replace NaNs/Infs
                # Still return the last valid state if available and better
                if np.any(np.isfinite(self.last_raw_state)):
                    return self.last_raw_state
                else: # If last state was also invalid, return the cleaned current one
                    self.last_raw_state = state.copy()
                    return state

            self.last_raw_state = state.copy() # Store the latest valid raw state

        except traci.exceptions.TraCIException as e:
            # Only print warning if it's not the common "vehicle not known" error after disappearance
            if "Vehicle '" + ego_id + "' is not known" not in str(e):
                 print(f"Warning: TraCI Error getting raw state for {ego_id}: {e}. Returning last known raw state.")
            return self.last_raw_state # Return last state on TraCI error
        except Exception as e:
            print(f"Warning: Unknown error getting raw state for {ego_id}: {e}. Returning last known raw state.")
            traceback.print_exc()
            return self.last_raw_state

        return self.last_raw_state

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """Perform action, return (next_raw_state, done) - NO CHANGES NEEDED"""
        done = False
        ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            # print(f"Debug: Ego '{ego_id}' not found at start of step {self.current_step}. Returning done=True.") # Debug
            self.collision_occurred = True # Assume collision/despawn if gone unexpectedly
            return self.last_raw_state, True # Return last known state, done=True

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id)
            num_lanes = self.num_lanes
            current_lane = np.clip(current_lane, 0, num_lanes - 1)

            # --- Execute Lane Change Command if requested ---
            # The decision logic is outside, here we just execute
            if action == 1 and current_lane > 0:
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0) # Request change left
            elif action == 2 and current_lane < (num_lanes - 1):
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0) # Request change right
            # If action is 0 (Keep Lane), do nothing specific here, just step below

            # --- Simulation Step ---
            traci.simulationStep()
            self.current_step += 1

            # --- Check Collision AFTER step ---
            # Use TraCI's collision detection
            collisions = traci.simulation.getCollisions()
            ego_collided_explicitly = False
            for col in collisions:
                # Check if our ego vehicle was involved either as collider or victim
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True
                    ego_collided_explicitly = True
                    done = True
                    # print(f"Debug: Collision detected involving {ego_id} at step {self.current_step}. Details: {col}") # Debug
                    break # No need to check further collisions

            # --- Check if Ego Still Exists AFTER step ---
            ego_exists_after_step = ego_id in traci.vehicle.getIDList()

            # If ego doesn't exist AND no explicit collision was logged, assume it finished or despawned safely/problematically
            if not ego_exists_after_step and not ego_collided_explicitly:
                # print(f"Debug: Ego '{ego_id}' disappeared at step {self.current_step} without explicit collision log.") # Debug
                done = True
                # We might not mark this as a collision unless it's unexpected. Depends on scenario.
                # For safety, let's assume disappearance without collision is 'done'.

            # --- Get Next State ---
            if ego_exists_after_step:
                next_raw_state = self._get_raw_state()
            else:
                # If ego is gone, return the last known state before it disappeared
                next_raw_state = self.last_raw_state

            # --- Check Other Termination Conditions ---
            # Simulation time limit (e.g., 1 hour)
            if traci.simulation.getTime() >= 3600:
                print("Debug: Simulation time limit reached.") # Debug
                done = True
            # Max steps per episode limit
            if self.current_step >= self.eval_config.EVAL_MAX_STEPS:
                # print(f"Debug: Max steps ({self.eval_config.EVAL_MAX_STEPS}) reached.") # Debug
                done = True

        except traci.exceptions.TraCIException as e:
            # Handle TraCI errors during the step
            is_not_known_error = "Vehicle '" + ego_id + "' is not known" in str(e)
            # If the error is 'not known', it likely means the vehicle disappeared (crashed/finished)
            # If it's another TraCI error, log it more verbosely.
            if not is_not_known_error:
                 print(f"Error: TraCI Exception during step {self.current_step}: {e}")

            # Assume collision if TraCI error occurs, unless it was just 'not known' and no collision was flagged
            if is_not_known_error and not self.collision_occurred:
                 pass # Vehicle disappeared, already handled by done=True check above potentially
            else:
                 self.collision_occurred = True # Mark collision for other errors or if already flagged

            done = True # End episode on TraCI error
            next_raw_state = self.last_raw_state # Return last known state
        except Exception as e:
            # Handle unexpected Python errors during the step
            print(f"Error: Unknown Exception during step {self.current_step}: {e}")
            traceback.print_exc()
            done = True
            self.collision_occurred = True # Assume collision on unknown error
            next_raw_state = self.last_raw_state # Return last known state

        return next_raw_state, done

    def get_vehicle_info(self):
        """Get current info like speed, lane, position, distance traveled, front distance"""
        # --- NO CHANGES NEEDED ---
        ego_id = self.ego_vehicle_id
        if ego_id in traci.vehicle.getIDList():
            try:
                speed = traci.vehicle.getSpeed(ego_id)
                lane = traci.vehicle.getLaneIndex(ego_id)
                pos = traci.vehicle.getPosition(ego_id)
                self.last_ego_pos = pos # Update last known position
                # Calculate distance from start if start position is known
                dist_traveled = math.dist(pos, self.ego_start_pos) if self.ego_start_pos else 0.0
                # Get front distance from the *last updated raw state* (index 2)
                # Use max_distance as default if state is invalid or not populated yet
                front_dist = self.last_raw_state[2] if len(self.last_raw_state) > 2 and np.isfinite(self.last_raw_state[2]) else self.dqn_train_config.max_distance
                front_dist = max(0, front_dist) # Ensure non-negative

                return {"speed": speed, "lane": lane, "pos": pos, "dist": dist_traveled, "front_dist": front_dist}
            except traci.exceptions.TraCIException:
                 # Error getting info, return info based on last known position/state
                 dist_traveled = math.dist(self.last_ego_pos, self.ego_start_pos) if self.last_ego_pos and self.ego_start_pos else 0.0
                 front_dist = self.last_raw_state[2] if len(self.last_raw_state) > 2 and np.isfinite(self.last_raw_state[2]) else self.dqn_train_config.max_distance
                 front_dist = max(0, front_dist)
                 return {"speed": 0, "lane": -1, "pos": self.last_ego_pos, "dist": dist_traveled, "front_dist": front_dist}
        else:
            # Vehicle doesn't exist, return info based on last known position/state
            dist_traveled = math.dist(self.last_ego_pos, self.ego_start_pos) if self.last_ego_pos and self.ego_start_pos else 0.0
            front_dist = self.last_raw_state[2] if len(self.last_raw_state) > 2 and np.isfinite(self.last_raw_state[2]) else self.dqn_train_config.max_distance
            front_dist = max(0, front_dist)
            return {"speed": 0, "lane": -1, "pos": self.last_ego_pos, "dist": dist_traveled, "front_dist": front_dist}

    def _close(self):
        """Close SUMO instance - NO CHANGES NEEDED"""
        if self.sumo_process:
            try: traci.close()
            except Exception: pass # Ignore errors on close
            finally:
                try:
                    if self.sumo_process.poll() is None: # Check if process is still running
                         self.sumo_process.terminate()
                         self.sumo_process.wait(timeout=2) # Wait for graceful termination
                except subprocess.TimeoutExpired: # If it doesn't terminate gracefully
                     print("Warning: SUMO process did not terminate gracefully. Killing.")
                     self.sumo_process.kill()
                     self.sumo_process.wait(timeout=1) # Wait for kill
                except Exception as e:
                     print(f"Warning: Error during SUMO termination: {e}")
                self.sumo_process = None
                self.traci_port = None
                time.sleep(0.1) # Short pause after closing
        else:
             self.traci_port = None # Ensure port is cleared even if process was already None

#####################################
# 模型加载和动作选择                   #
#####################################

# --- load_model: NO CHANGES NEEDED ---
def load_model(model_path: str, model_type: str, train_config: Any, device: torch.device) -> nn.Module:
    """Load a trained model (DQN or PPO)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type == 'dqn':
        if not isinstance(train_config, DQN_Config_Base):
            raise TypeError(f"Provided config for DQN model is not instance of DQN_Config_Base, got {type(train_config)}")
        # Ensure train_config has necessary attributes (state_dim, action_dim, hidden_size, etc.)
        model = QNetwork(train_config.state_dim, train_config.action_dim, train_config.hidden_size, train_config).to(device)
        strict_loading = True # DQN usually requires strict loading
    elif model_type == 'ppo':
        if not isinstance(train_config, PPO_Config_Base):
             raise TypeError(f"Provided config for PPO model is not instance of PPO_Config_Base, got {type(train_config)}")
        model = PPO(train_config.state_dim, train_config.action_dim, train_config.hidden_size).to(device)
        # PPO might have value head separate, allow non-strict loading if necessary
        # However, for evaluation we usually load the actor part which should be consistent.
        # Let's try strict first, change if loading fails.
        strict_loading = False # PPO often has extra keys (value head, log_std) not needed for pure actor eval
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        # print(f"Loading model state_dict from: {model_path} (Strict={strict_loading})") # Debug
        model.load_state_dict(torch.load(model_path, map_location=device), strict=strict_loading)
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded {model_type.upper()} model from: {os.path.basename(model_path)}")
        return model
    except RuntimeError as e:
         print(f"RuntimeError loading state_dict from {model_path} (Strict={strict_loading}): {e}")
         if model_type == 'ppo' and strict_loading:
             print("Retrying PPO load with strict=False")
             try:
                 model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                 model.eval()
                 print(f"Successfully loaded PPO model with strict=False.")
                 return model
             except Exception as e_retry:
                  print(f"Error loading PPO model even with strict=False: {e_retry}")
                  raise e_retry
         else:
              raise e # Reraise original error if not PPO or retry failed
    except FileNotFoundError:
         print(f"Error: Model file not found at {model_path}")
         raise
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        raise

# --- normalize_state: NO CHANGES NEEDED ---
def normalize_state(state_raw: np.ndarray, normalizer: Optional[RunningMeanStd], clip_val: float) -> np.ndarray:
    """Normalize state using the provided FROZEN normalizer instance"""
    if normalizer:
        # Use the stored mean and std, DO NOT UPDATE during evaluation
        mean = normalizer.mean
        std = normalizer.std + 1e-8 # Add epsilon for numerical stability

        # Ensure dimensions match for broadcasting if state_raw is single instance
        if state_raw.ndim == 1 and mean.ndim == 1 and state_raw.shape[0] == mean.shape[0]:
            norm_state = (state_raw - mean) / std
        elif state_raw.ndim == 2 and mean.ndim == 1 and state_raw.shape[1] == mean.shape[0]:
            # Batch normalization
            norm_state = (state_raw - mean[np.newaxis, :]) / std[np.newaxis, :]
        else:
             print(f"Warning: Shape mismatch in normalize_state. Raw state shape: {state_raw.shape}, Normalizer mean shape: {mean.shape}. Returning raw state.")
             return state_raw.astype(np.float32)

        norm_state = np.clip(norm_state, -clip_val, clip_val)
        return norm_state.astype(np.float32)
    else:
        # If no normalizer, return the raw state as float32
        return state_raw.astype(np.float32)

# --- get_dqn_action: NO CHANGES NEEDED ---
# This function should return the agent's *intended* action based on its policy
def get_dqn_action(model: QNetwork, state_norm: np.ndarray, current_lane_idx: int, config: DQN_Config_Base, num_eval_lanes: int, device: torch.device) -> int:
    """Get action from DQN model (C51/Noisy aware) based on its policy"""
    model.eval() # Ensure model is in eval mode

    # Reset noise for NoisyNets if used
    if config.use_noisy_nets:
        model.reset_noise()

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        # Get action probability distributions from the network
        action_probs = model(state_tensor) # Shape: [1, action_dim, num_atoms]

        # Calculate expected Q-values from the distribution for action selection
        support = torch.linspace(config.v_min, config.v_max, config.num_atoms).to(device)
        expected_q_values = (action_probs * support).sum(dim=2) # Shape: [1, action_dim]

        # --- Mask illegal actions based on current lane ---
        q_values_masked = expected_q_values.clone()
        # Ensure current_lane_idx is valid
        current_lane_idx = np.clip(current_lane_idx, 0, num_eval_lanes - 1)

        # Action 1: Change Left is illegal if in the leftmost lane (index 0)
        if current_lane_idx == 0:
            q_values_masked[0, 1] = -float('inf')
        # Action 2: Change Right is illegal if in the rightmost lane
        if current_lane_idx >= num_eval_lanes - 1:
            q_values_masked[0, 2] = -float('inf')

        # Select the action with the highest Q-value after masking
        action = q_values_masked.argmax().item()

    return action

# --- get_ppo_action: NO CHANGES NEEDED ---
# This function should return the agent's *intended* action based on its policy
def get_ppo_action(model: PPO, state_norm: np.ndarray, current_lane_idx: int, num_eval_lanes: int, device: torch.device) -> int:
    """Get deterministic action from PPO actor based on its policy"""
    model.eval() # Ensure model is in eval mode

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        # Get action probabilities from the PPO actor's policy head
        action_probs = model.get_action_probs(state_tensor) # Shape: [1, action_dim]

        # --- Mask illegal actions ---
        probs_masked = action_probs.clone()
        current_lane_idx = np.clip(current_lane_idx, 0, num_eval_lanes - 1)

        # Action 1 (Left): Set probability to 0 if in leftmost lane
        if current_lane_idx == 0:
            probs_masked[0, 1] = 0.0
        # Action 2 (Right): Set probability to 0 if in rightmost lane
        if current_lane_idx >= num_eval_lanes - 1:
            probs_masked[0, 2] = 0.0

        # Renormalize probabilities after masking (important!)
        probs_sum = probs_masked.sum(dim=-1, keepdim=True)
        if probs_sum.item() > 1e-8: # Avoid division by zero if all actions masked (shouldn't happen if keep lane possible)
            final_probs = probs_masked / probs_sum
        else:
            # If somehow all actions are masked, default to keep lane (action 0)
            final_probs = torch.zeros_like(probs_masked)
            final_probs[0, 0] = 1.0

        # Choose the action with the highest probability (deterministic selection for evaluation)
        action = final_probs.argmax().item()

    return action


#####################################
# 评估回合运行器                    #
#####################################

# --- Updated Episode Result ---
EpisodeResult = namedtuple('EpisodeResult', [
    'steps', 'collided', 'avg_speed', 'total_dist',
    'forced_attempts',          # How many times script proposed a change
    'forced_agreed',            # How many times agent's policy matched proposal
    'forced_disagreed',         # **** NEW: How many times agent policy mismatched ****
    'forced_executed_safe',     # How many *agreed* changes completed safely
    'forced_executed_collision',# How many *agreed* changes resulted in collision
    'agreed_lane_changes',      # How many lane changes were actually executed (should == forced_agreed)
    'min_front_dist'
])

# --- Updated Evaluate Episode Return Type ---
EvaluateEpisodeReturnType = Tuple[
    EpisodeResult,
    List[float], List[int], List[int], List[float] # Speeds, Lanes, Actions, Front Dists
]

# --- evaluate_episode: NO CHANGES NEEDED for color fix ---
def evaluate_episode(
    model: nn.Module,
    model_type: str, # 'dqn' or 'ppo'
    env: EvaluationEnv,
    train_config: Any, # DQN_Config_Base or PPO_Config_Base
    obs_normalizer: Optional[RunningMeanStd], # Initial (frozen) normalizer state
    device: torch.device,
    eval_config: EvalConfig
) -> EvaluateEpisodeReturnType:
    """
    Run a single evaluation episode.
    MODIFIED: Agent only changes lane if explicitly proposed by the script AND agent agrees.
              Otherwise, action 0 (keep lane) is forced.
    """

    state_raw = env.reset() # Returns raw state
    if np.any(np.isnan(state_raw)):
         print("Error: NaN detected in initial state after reset. Aborting episode.")
         # Return a dummy result indicating failure
         dummy_result = EpisodeResult(0, True, 0, 0, 0, 0, 0, 0, 0, 0, env.dqn_train_config.max_distance)
         return dummy_result, [], [], [], []

    current_obs_normalizer = obs_normalizer # Use frozen normalizer

    done = False
    step_count = 0
    agreed_lane_changes = 0 # Renamed from model_lane_changes, only counts executed agreed changes
    min_front_dist_episode = env.dqn_train_config.max_distance

    # --- Lists to collect step-level data ---
    all_speeds_in_episode: List[float] = []
    all_lanes_in_episode: List[int] = []
    all_actions_in_episode: List[int] = [] # Stores the *executed* action
    all_front_distances_in_episode: List[float] = []

    # --- Forced Change Tracking ---
    forced_attempts = 0         # Times script proposed a change
    forced_agreed = 0           # Times agent policy matched proposal
    forced_disagreed = 0        # **** NEW: Times agent policy mismatched proposal ****
    forced_executed_safe = 0    # *Agreed* changes completed safely
    forced_executed_collision = 0 # *Agreed* changes caused collision

    # --- Monitoring state for AGREED changes ---
    monitoring_change = False     # Is an agreed change being monitored?
    monitor_steps_left = 0      # How many steps left for monitoring
    monitor_target_action = -1  # Which change action is being monitored (1 or 2)
    monitor_start_lane = -1     # Lane index when monitoring started
    monitor_start_pos = None    # Position when monitoring started

    last_valid_vehicle_info = env.get_vehicle_info() # Get initial info

    while not done and step_count < eval_config.EVAL_MAX_STEPS:
        # 1. Get Normalized State
        clip_value = train_config.obs_norm_clip if hasattr(train_config, 'obs_norm_clip') else 5.0
        state_norm = normalize_state(state_raw, current_obs_normalizer, clip_value)
        if np.any(np.isnan(state_norm)):
             print(f"Error: NaN detected in normalized state at step {step_count}. Aborting episode.")
             done = True; env.collision_occurred = True # Mark as error/collision
             break

        # 2. Determine Current Lane and Possible Changes
        # Use the raw state's lane info (index 1) for decision making
        if not np.any(np.isnan(state_raw)) and len(state_raw) > 1:
             current_lane_idx = int(round(state_raw[1]))
             current_lane_idx = np.clip(current_lane_idx, 0, eval_config.NUM_LANES - 1)
        else: # Fallback to last known valid info if raw state is bad
             current_lane_idx = last_valid_vehicle_info['lane'] if last_valid_vehicle_info['lane'] >= 0 else 0
             current_lane_idx = np.clip(current_lane_idx, 0, eval_config.NUM_LANES - 1)

        can_go_left = current_lane_idx > 0
        can_go_right = current_lane_idx < (eval_config.NUM_LANES - 1)

        # 3. Decide Final Action based on Forced Change Logic
        final_action_to_execute = 0 # Default: Keep Lane
        proposed_forced_action = -1 # Reset proposal for this step

        # Check if monitoring a previous *agreed* attempt
        if monitoring_change:
            monitor_steps_left -= 1
            current_vehicle_info = env.get_vehicle_info() # Check state before stepping again
            current_pos = current_vehicle_info['pos']
            current_lane_after_step = current_vehicle_info['lane'] # Current lane *now*

            lateral_dist = 0.0
            if current_pos and monitor_start_pos:
                 lateral_dist = abs(current_pos[1] - monitor_start_pos[1])

            # Check if change completed based on lane index or sufficient lateral movement
            lane_changed_logically = (current_lane_after_step >= 0 and current_lane_after_step != monitor_start_lane)
            # change_succeeded_criteria = lane_changed_logically # Require lane index change
            # Use lateral distance as criterion
            change_succeeded_criteria = lateral_dist >= eval_config.FORCE_CHANGE_SUCCESS_DIST

            # Check if a collision happened *during* monitoring
            if env.collision_occurred: # Collision detected by env.step in previous iteration
                # print(f"Debug: Collision occurred during monitoring window for action {monitor_target_action}") # Debug
                forced_executed_collision += 1
                monitoring_change = False # Stop monitoring due to collision
                done = True # End episode on collision

            elif change_succeeded_criteria:
                # print(f"Debug: Agreed change {monitor_target_action} successful (Lateral dist: {lateral_dist:.2f} >= {eval_config.FORCE_CHANGE_SUCCESS_DIST}, Lane {monitor_start_lane}->{current_lane_after_step}).") # Debug
                forced_executed_safe += 1
                monitoring_change = False # Stop monitoring, success

            elif monitor_steps_left <= 0:
                # print(f"Debug: Agreed change {monitor_target_action} timed out (Monitor steps expired, lateral dist: {lateral_dist:.2f}).") # Debug
                monitoring_change = False # Stop monitoring, timeout/failed

            # If still monitoring, the action for *this* step must be keep lane
            if monitoring_change:
                 final_action_to_execute = 0

        # If NOT monitoring a change, check if it's time to PROPOSE a new one
        if not monitoring_change:
            is_forced_attempt_step = (step_count > 0 and step_count % eval_config.FORCE_CHANGE_INTERVAL_STEPS == 0)

            if is_forced_attempt_step:
                # Propose a change if possible
                possible_proposals = []
                if can_go_left: possible_proposals.append(1)
                if can_go_right: possible_proposals.append(2)

                if possible_proposals:
                    proposed_forced_action = random.choice(possible_proposals)
                    forced_attempts += 1
                    # print(f"Debug: Step {step_count}: Proposing forced action {proposed_forced_action} (current lane {current_lane_idx})") # Debug

                    # Get the agent's *intended* action for this state
                    if model_type == 'dqn':
                        intended_action = get_dqn_action(model, state_norm, current_lane_idx, train_config, eval_config.NUM_LANES, device)
                    elif model_type == 'ppo':
                        intended_action = get_ppo_action(model, state_norm, current_lane_idx, eval_config.NUM_LANES, device)
                    else:
                        intended_action = 0
                    # print(f"Debug: Step {step_count}: Agent intended action {intended_action}") # Debug

                    # Compare proposal and intention
                    if intended_action == proposed_forced_action:
                        forced_agreed += 1
                        final_action_to_execute = intended_action # Execute the agreed change
                        agreed_lane_changes += 1 # Increment count of executed changes
                        # print(f"Debug: Step {step_count}: Agent AGREED. Executing {final_action_to_execute}. Starting monitoring.") # Debug

                        # Start monitoring this agreed change
                        monitoring_change = True
                        monitor_steps_left = eval_config.FORCE_CHANGE_MONITOR_STEPS
                        monitor_target_action = final_action_to_execute
                        monitor_start_lane = current_lane_idx
                        # Get position *before* executing the change action
                        start_info = env.get_vehicle_info()
                        monitor_start_pos = start_info['pos'] if start_info['lane'] != -1 else None

                    else: # Agent disagreed
                        forced_disagreed += 1
                        final_action_to_execute = 0 # Force Keep Lane
                        # print(f"Debug: Step {step_count}: Agent DISAGREED (intended {intended_action}, proposed {proposed_forced_action}). Forcing action 0.") # Debug
                else:
                    # Cannot propose a change (e.g., no valid adjacent lanes)
                    final_action_to_execute = 0 # Keep Lane
                    # print(f"Debug: Step {step_count}: Cannot propose forced change. Forcing action 0.") # Debug

            else: # Not a forced attempt step
                final_action_to_execute = 0 # Force Keep Lane
                # Get agent's intended action only for potential (unused) analysis if needed
                # if model_type == 'dqn': intended_action = get_dqn_action(...)
                # else: intended_action = get_ppo_action(...)
                # if intended_action != 0: # Agent wanted to change but wasn't allowed
                #     pass # Optionally track this

        # 4. Store the *executed* action
        all_actions_in_episode.append(final_action_to_execute)

        # 5. Step the environment with the determined action
        if done: break # Exit loop if done flag was set earlier (e.g., by monitoring collision)
        next_state_raw, done = env.step(final_action_to_execute)

        # 6. Get info AFTER step & store data
        state_raw = next_state_raw # Update state for next iteration
        step_count += 1
        vehicle_info = env.get_vehicle_info() # Contains speed, lane, front_dist after step

        if vehicle_info['lane'] != -1: # Store data only if vehicle exists
            all_speeds_in_episode.append(vehicle_info['speed'])
            all_lanes_in_episode.append(vehicle_info['lane'])
            current_front_dist = vehicle_info['front_dist'] if vehicle_info['front_dist'] is not None and np.isfinite(vehicle_info['front_dist']) and vehicle_info['front_dist'] >= 0 else env.dqn_train_config.max_distance
            all_front_distances_in_episode.append(current_front_dist)
            min_front_dist_episode = min(min_front_dist_episode, current_front_dist)
            last_valid_vehicle_info = vehicle_info # Update last valid info
        else: # Vehicle doesn't exist (finished or crashed without explicit flag in env.step)
            all_speeds_in_episode.append(0)
            all_lanes_in_episode.append(-1)
            all_front_distances_in_episode.append(env.dqn_train_config.max_distance)
            if not done: # If env.step didn't set done, set it now as vehicle is gone
                 done = True
                 # print(f"Debug: Setting done=True because vehicle info has lane=-1 at step {step_count}.") # Debug

        # If env.step set collision flag, ensure done is True
        if env.collision_occurred and not done:
             done = True
             # print(f"Debug: Setting done=True because env.collision_occurred is True at end of step {step_count}.") # Debug


    # --- Episode End ---
    # Final check on vehicle info if episode ended normally
    if not last_valid_vehicle_info or last_valid_vehicle_info['lane'] == -1:
         last_valid_vehicle_info = env.get_vehicle_info()

    avg_speed = np.mean([s for s in all_speeds_in_episode if s is not None]) if all_speeds_in_episode else 0.0
    total_dist = last_valid_vehicle_info['dist'] if last_valid_vehicle_info else 0.0
    collided = env.collision_occurred

    result = EpisodeResult(
        steps=step_count,
        collided=collided,
        avg_speed=avg_speed,
        total_dist=total_dist,
        forced_attempts=forced_attempts,
        forced_agreed=forced_agreed,
        forced_disagreed=forced_disagreed, # Include new metric
        forced_executed_safe=forced_executed_safe,
        forced_executed_collision=forced_executed_collision,
        agreed_lane_changes=agreed_lane_changes, # Include renamed metric
        min_front_dist=min_front_dist_episode
    )

    # Return episode summary AND step-level data lists
    return result, all_speeds_in_episode, all_lanes_in_episode, all_actions_in_episode, all_front_distances_in_episode


#####################################
# 主评估脚本                       #
#####################################
def main():
    eval_config = EvalConfig()
    # Load base configs to get state_dim etc.
    try: dqn_train_config = DQN_Config_Base()
    except Exception as e: print(f"Error loading DQN_Config_Base: {e}"); sys.exit(1)
    try: ppo_train_config = PPO_Config_Base()
    except Exception as e: print(f"Error loading PPO_Config_Base: {e}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evaluation_results_forced_only_{timestamp}"; os.makedirs(output_dir, exist_ok=True); print(f"Output will be saved in: {output_dir}")

    # --- File Checks ---
    print("\n--- Checking Files ---")
    if not os.path.exists(eval_config.DQN_MODEL_PATH): print(f"ERROR: DQN model path not found: {eval_config.DQN_MODEL_PATH}"); sys.exit(1)
    else: print(f"Found DQN model: {eval_config.DQN_MODEL_PATH}")
    if not os.path.exists(eval_config.PPO_MODEL_PATH): print(f"ERROR: PPO model path not found: {eval_config.PPO_MODEL_PATH}"); sys.exit(1)
    else: print(f"Found PPO model: {eval_config.PPO_MODEL_PATH}")
    if not os.path.exists(eval_config.EVAL_SUMO_CONFIG): print(f"ERROR: Evaluation SUMO config not found: {eval_config.EVAL_SUMO_CONFIG}"); sys.exit(1)
    else: print(f"Found SUMO config: {eval_config.EVAL_SUMO_CONFIG}")
    try:
        with open(eval_config.EVAL_SUMO_CONFIG, 'r') as f: content = f.read()
        # Update checks for new file names if they differ
        if 'net-file value="new.net.xml"' not in content: print("Warning: 'new.net.xml' not found within evaluation sumocfg!")
        if 'route-files value="new.rou.xml"' not in content: print("Warning: 'new.rou.xml' not found within evaluation sumocfg!")
        if not os.path.exists("new.net.xml"): print("Warning: File new.net.xml not found.")
        if not os.path.exists("new.rou.xml"): print("Warning: File new.rou.xml not found.")
        print("SUMO config file checks passed (warnings are informational).")
    except Exception as e: print(f"Warning: Could not read evaluation SUMO config {eval_config.EVAL_SUMO_CONFIG}: {e}")

    # --- Load Models ---
    print("\n--- Loading Models ---")
    dqn_model = load_model(eval_config.DQN_MODEL_PATH, 'dqn', dqn_train_config, device)
    ppo_model = load_model(eval_config.PPO_MODEL_PATH, 'ppo', ppo_train_config, device)

    # --- Initialize Normalizers (FROZEN) ---
    # IMPORTANT: If normalization was used during training, the *trained*
    # normalizer stats (mean, std) should be loaded here.
    # For now, we recreate them based on config flags, assuming they start from zero
    # and are NOT updated during evaluation.
    dqn_obs_normalizer = RunningMeanStd(shape=(dqn_train_config.state_dim,)) if dqn_train_config.normalize_observations else None
    ppo_obs_normalizer = RunningMeanStd(shape=(ppo_train_config.state_dim,)) if ppo_train_config.normalize_observations else None
    if dqn_obs_normalizer or ppo_obs_normalizer:
        print("Initializing (frozen) normalizers for evaluation based on train config flags.")
        print("NOTE: Ideally, load saved training normalizer stats (mean/std) here.")
        # Example placeholder for loading stats (replace with actual loading code if available)
        # try:
        #     dqn_norm_stats = torch.load("dqn_normalizer_stats.pth") # Example path
        #     dqn_obs_normalizer.mean = dqn_norm_stats['mean']
        #     dqn_obs_normalizer.var = dqn_norm_stats['var']
        #     print("Loaded DQN normalizer stats.")
        # except FileNotFoundError:
        #     print("DQN normalizer stats file not found, using initial zero stats.")
        # except Exception as e:
        #     print(f"Error loading DQN normalizer stats: {e}. Using initial zero stats.")
        # (Repeat for PPO if needed)

    # --- Initialize Environment ---
    base_seed = eval_config.EVAL_SEED
    # Pass DQN config for state dim etc., actual model config used inside evaluate_episode
    env = EvaluationEnv(eval_config, dqn_train_config, base_seed)

    # --- Run Evaluation ---
    print(f"\n--- Running Evaluation ({eval_config.EVAL_EPISODES} episodes/model) ---")
    print(f"--- MODE: Lane changes ONLY when proposed AND agreed by agent ---")
    dqn_results_list: List[EpisodeResult] = []
    ppo_results_list: List[EpisodeResult] = []
    # Lists to store step-level data across ALL episodes for each model
    dqn_all_speeds: List[float] = []
    dqn_all_lanes: List[int] = []
    dqn_all_actions: List[int] = [] # Executed actions
    dqn_all_front_dists: List[float] = []
    ppo_all_speeds: List[float] = []
    ppo_all_lanes: List[int] = []
    ppo_all_actions: List[int] = [] # Executed actions
    ppo_all_front_dists: List[float] = []

    evaluation_successful = True # Flag to track if evaluation completed without major errors
    for i in tqdm(range(eval_config.EVAL_EPISODES), desc="Evaluating Episodes"):
        episode_seed = base_seed + i
        print(f"\n--- Episode {i+1}/{eval_config.EVAL_EPISODES} (Seed: {episode_seed}) ---")

        # --- Evaluate DQN ---
        print("Evaluating DQN...")
        env.sumo_seed = episode_seed # Set seed for this episode run
        try:
            dqn_result, speeds, lanes, actions, f_dists = evaluate_episode(
                dqn_model, 'dqn', env, dqn_train_config, dqn_obs_normalizer, device, eval_config
            )
            if dqn_result.steps == 0 and dqn_result.collided: # Check for immediate failure from reset/init
                 print("DQN evaluation episode failed immediately. Skipping result.")
            else:
                 dqn_results_list.append(dqn_result)
                 dqn_all_speeds.extend(speeds)
                 dqn_all_lanes.extend(lanes)
                 dqn_all_actions.extend(actions)
                 dqn_all_front_dists.extend(f_dists)
                 print(f"DQN Ep {i+1} Result: Steps={dqn_result.steps}, Collided={dqn_result.collided}, AgreedLC={dqn_result.agreed_lane_changes}, Disagreed={dqn_result.forced_disagreed}")
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
            print(f"\nError during DQN evaluation episode {i+1}: {e}")
            print("Attempting to close environment and continue...")
            evaluation_successful = False
            time.sleep(0.5); env._close(); time.sleep(1); continue # Skip to next episode
        except Exception as e_other:
            print(f"\nUnexpected error during DQN evaluation episode {i+1}: {e_other}")
            traceback.print_exc()
            evaluation_successful = False
            print("Attempting to close environment and continue...")
            time.sleep(0.5); env._close(); time.sleep(1); continue # Skip to next episode

        # --- Evaluate PPO ---
        print("Evaluating PPO...")
        env.sumo_seed = episode_seed # Use the same seed for PPO for fair comparison
        try:
            ppo_result, speeds, lanes, actions, f_dists = evaluate_episode(
                ppo_model, 'ppo', env, ppo_train_config, ppo_obs_normalizer, device, eval_config
            )
            if ppo_result.steps == 0 and ppo_result.collided:
                 print("PPO evaluation episode failed immediately. Skipping result.")
            else:
                ppo_results_list.append(ppo_result)
                ppo_all_speeds.extend(speeds)
                ppo_all_lanes.extend(lanes)
                ppo_all_actions.extend(actions)
                ppo_all_front_dists.extend(f_dists)
                print(f"PPO Ep {i+1} Result: Steps={ppo_result.steps}, Collided={ppo_result.collided}, AgreedLC={ppo_result.agreed_lane_changes}, Disagreed={ppo_result.forced_disagreed}")
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
            print(f"\nError during PPO evaluation episode {i+1}: {e}")
            print("Attempting to close environment and continue...")
            evaluation_successful = False
            time.sleep(0.5); env._close(); time.sleep(1); continue
        except Exception as e_other:
            print(f"\nUnexpected error during PPO evaluation episode {i+1}: {e_other}")
            traceback.print_exc()
            evaluation_successful = False
            print("Attempting to close environment and continue...")
            time.sleep(0.5); env._close(); time.sleep(1); continue

    env._close() # Close env after all episodes
    print("\n--- Evaluation Finished ---")

    # --- Aggregate and Compare Results ---
    if not dqn_results_list and not ppo_results_list:
        print("No evaluation results collected. Exiting.")
        return
    if not evaluation_successful:
         print("WARNING: Evaluation encountered errors. Results might be incomplete or inaccurate.")

    results = {'dqn': {}, 'ppo': {}}
    # --- Define Metrics (including new ones) ---
    # (key, name, format, higher_is_better, std_dev_key, median_key)
    metrics_definitions = [
        ('total_episodes', '总评估回合数', 'd', True, None, None),
        ('avg_steps', '平均步数/回合', '.1f', True, 'std_steps', 'median_steps'),
        ('std_steps', '步数标准差', '.1f', False, None, None),
        ('median_steps', '步数中位数', '.1f', True, None, None),
        ('avg_speed_episode', '每回合平均速度 (m/s)', '.2f', True, 'std_speed_episode', None), # Avg of episode averages
        ('median_speed_all', '所有步速度中位数 (m/s)', '.2f', True, None, None), # Median of all steps
        ('avg_dist', '平均距离/回合 (m)', '.1f', True, 'std_dist', 'median_dist'),
        ('std_dist', '距离标准差 (m)', '.1f', False, None, None),
        ('median_dist', '距离中位数 (m)', '.1f', True, None, None),
        ('collision_rate', '碰撞率 (%)', '.1f', False, None, None),
        ('median_min_front_dist', '最小前车距离中位数 (m)', '.1f', True, None, None),
        ('avg_agreed_lc', '平均同意换道次数/回合', '.1f', None, None, None), # Avg agreed LCs
        ('total_forced_attempts', '强制换道尝试总数', 'd', None, None, None),
        ('forced_agreement_rate', '强制换道: 同意率 (%)', '.1f', True, None, None), # agreed / attempts
        ('forced_disagreement_rate', '强制换道: 不同意率 (%)', '.1f', False, None, None), # disagreed / attempts
        ('forced_execution_success_rate_agreed', '执行成功率 (% 同意的)', '.1f', True, None, None), # safe / agreed
        ('forced_execution_collision_rate_agreed', '执行碰撞率 (% 同意的)', '.1f', False, None, None), # collision / agreed
    ]


    # --- Data Aggregation Loop ---
    for model_key, results_list, all_speeds, all_lanes, all_actions, all_f_dists in [
            ('dqn', dqn_results_list, dqn_all_speeds, dqn_all_lanes, dqn_all_actions, dqn_all_front_dists),
            ('ppo', ppo_results_list, ppo_all_speeds, ppo_all_lanes, ppo_all_actions, ppo_all_front_dists)]:

        total_episodes = len(results_list)
        results[model_key]['total_episodes'] = total_episodes
        if total_episodes == 0:
            print(f"Warning: No results collected for {model_key.upper()}. Skipping aggregation.")
            continue

        # Episode-level stats
        steps_list = [r.steps for r in results_list]
        speeds_avg_list = [r.avg_speed for r in results_list] # List of average speeds per episode
        dists_list = [r.total_dist for r in results_list]
        min_front_dist_list = [r.min_front_dist for r in results_list]
        collided_list = [r.collided for r in results_list]
        agreed_lc_list = [r.agreed_lane_changes for r in results_list] # Use the new metric name

        results[model_key]['avg_steps'] = np.mean(steps_list); results[model_key]['std_steps'] = np.std(steps_list); results[model_key]['median_steps'] = np.median(steps_list)
        results[model_key]['collision_rate'] = np.mean(collided_list) * 100
        results[model_key]['avg_speed_episode'] = np.mean(speeds_avg_list); results[model_key]['std_speed_episode'] = np.std(speeds_avg_list);
        results[model_key]['median_speed_all'] = np.median(all_speeds) if all_speeds else 0 # Median of *all* step speeds
        results[model_key]['avg_dist'] = np.mean(dists_list); results[model_key]['std_dist'] = np.std(dists_list); results[model_key]['median_dist'] = np.median(dists_list)
        results[model_key]['avg_agreed_lc'] = np.mean(agreed_lc_list)
        results[model_key]['median_min_front_dist'] = np.median(min_front_dist_list) if min_front_dist_list else env.dqn_train_config.max_distance

        # Forced Change Aggregates
        total_forced_attempts = sum(r.forced_attempts for r in results_list)
        total_forced_agreed = sum(r.forced_agreed for r in results_list)
        total_forced_disagreed = sum(r.forced_disagreed for r in results_list) # Aggregate new metric
        total_forced_executed_safe = sum(r.forced_executed_safe for r in results_list)
        total_forced_executed_collision = sum(r.forced_executed_collision for r in results_list)

        results[model_key]['total_forced_attempts'] = total_forced_attempts
        results[model_key]['forced_agreement_rate'] = (total_forced_agreed / total_forced_attempts * 100) if total_forced_attempts > 0 else 0
        results[model_key]['forced_disagreement_rate'] = (total_forced_disagreed / total_forced_attempts * 100) if total_forced_attempts > 0 else 0
        # Rates based on *agreed* changes
        results[model_key]['forced_execution_success_rate_agreed'] = (total_forced_executed_safe / total_forced_agreed * 100) if total_forced_agreed > 0 else 0
        results[model_key]['forced_execution_collision_rate_agreed'] = (total_forced_executed_collision / total_forced_agreed * 100) if total_forced_agreed > 0 else 0

        # --- Calculate Step-Level Aggregates (Lane Occupancy, Action Distribution) ---
        results[model_key]['step_level_aggregates'] = {}
        # Lane Occupancy
        lane_counts = Counter(lane for lane in all_lanes if lane >= 0) # Count valid lanes
        total_valid_lane_steps = sum(lane_counts.values())
        lane_percentages = {lane: (count / total_valid_lane_steps * 100) if total_valid_lane_steps > 0 else 0
                            for lane, count in lane_counts.items()}
        results[model_key]['step_level_aggregates']['lane_occupancy_percent'] = {f"lane_{l}": lane_percentages.get(l, 0.0) for l in range(eval_config.NUM_LANES)}

        # Action Distribution (Based on *executed* actions)
        action_counts = Counter(all_actions)
        total_actions = len(all_actions)
        action_percentages = {action: (count / total_actions * 100) if total_actions > 0 else 0
                              for action, count in action_counts.items()}
        results[model_key]['step_level_aggregates']['action_distribution_percent'] = {f"action_{a}": action_percentages.get(a, 0.0) for a in range(3)} # Actions 0, 1, 2

    # --- Print and Save Text Comparison ---
    print("\n--- Results Comparison (Text) ---")
    comparison_lines = []
    header1 = f"{'Metric':<45} | {'DQN':<25} | {'PPO':<25}"
    header2 = "-" * len(header1)
    comparison_lines.append(header1); comparison_lines.append(header2)
    print(header1); print(header2)

    processed_keys = set() # To handle combined avg/median/std reporting

    for key, name, fmt, _, std_key, median_key in metrics_definitions:
        if key in processed_keys: continue

        dqn_val = results.get('dqn', {}).get(key, 'N/A')
        ppo_val = results.get('ppo', {}).get(key, 'N/A')

        # Format main value
        dqn_str = format(dqn_val, fmt) if isinstance(dqn_val, (int, float)) else str(dqn_val)
        ppo_str = format(ppo_val, fmt) if isinstance(ppo_val, (int, float)) else str(ppo_val)

        current_name = name # Use base name

        # Add Std Dev if applicable
        if std_key and std_key not in processed_keys:
            dqn_std = results.get('dqn', {}).get(std_key, None)
            ppo_std = results.get('ppo', {}).get(std_key, None)
            # Determine precision from the main format string (e.g., '.1f' -> 1)
            precision = int(fmt[-2]) if len(fmt) >= 2 and fmt[-1] == 'f' else 1
            if isinstance(dqn_std, (int, float)):
                dqn_str += f" (±{dqn_std:.{precision}f})"
            if isinstance(ppo_std, (int, float)):
                ppo_str += f" (±{ppo_std:.{precision}f})"
            processed_keys.add(std_key) # Mark std_dev as processed

        # Add Median if applicable (and different from the main key)
        elif median_key and median_key not in processed_keys:
            dqn_median = results.get('dqn', {}).get(median_key, 'N/A')
            ppo_median = results.get('ppo', {}).get(median_key, 'N/A')
            # Determine precision
            precision = int(fmt[-2]) if len(fmt) >= 2 and fmt[-1] == 'f' else 1
            dqn_median_str = format(dqn_median, f'.{precision}f') if isinstance(dqn_median, (int, float)) else 'N/A'
            ppo_median_str = format(ppo_median, f'.{precision}f') if isinstance(ppo_median, (int, float)) else 'N/A'

            # Combined approach (Avg (Med: Median) ± Std if std exists, else Avg (Med: Median))
            if std_key in processed_keys: # Check if std was already added
                dqn_str = dqn_str.replace(f" (±{dqn_std:.{precision}f})", f" (Med: {dqn_median_str} ±{dqn_std:.{precision}f})")
                ppo_str = ppo_str.replace(f" (±{ppo_std:.{precision}f})", f" (Med: {ppo_median_str} ±{ppo_std:.{precision}f})")
            else:
                dqn_str += f" (Med: {dqn_median_str})"
                ppo_str += f" (Med: {ppo_median_str})"
            processed_keys.add(median_key)


        line = f"{current_name:<45} | {dqn_str:<25} | {ppo_str:<25}"
        comparison_lines.append(line); print(line)
        processed_keys.add(key)


    # Add Lane Occupancy and Action Distribution to text summary
    comparison_lines.append("-" * len(header1))
    print("-" * len(header1))
    print("Step-Level Aggregates:")
    comparison_lines.append("Step-Level Aggregates:")

    for lane_i in range(eval_config.NUM_LANES):
         lane_key = f"lane_{lane_i}"
         name = f"  车道 {lane_i} 占用率 (%)"
         dqn_perc = results.get('dqn', {}).get('step_level_aggregates', {}).get('lane_occupancy_percent', {}).get(lane_key, 0.0)
         ppo_perc = results.get('ppo', {}).get('step_level_aggregates', {}).get('lane_occupancy_percent', {}).get(lane_key, 0.0)
         line = f"{name:<45} | {dqn_perc:.1f}%{'' :<20} | {ppo_perc:.1f}%{'' :<20}"
         comparison_lines.append(line); print(line)

    comparison_lines.append("-" * len(header1))
    print("-" * len(header1))
    action_names = {0: "保持车道 (强制/同意)", 1: "向左变道 (同意)", 2: "向右变道 (同意)"}
    for action_i in range(3):
         action_key = f"action_{action_i}"
         name = f"  执行动作 '{action_names[action_i]}' 选择率 (%)"
         dqn_perc = results.get('dqn', {}).get('step_level_aggregates', {}).get('action_distribution_percent', {}).get(action_key, 0.0)
         ppo_perc = results.get('ppo', {}).get('step_level_aggregates', {}).get('action_distribution_percent', {}).get(action_key, 0.0)
         line = f"{name:<45} | {dqn_perc:.1f}%{'' :<20} | {ppo_perc:.1f}%{'' :<20}"
         comparison_lines.append(line); print(line)

    comparison_lines.append(header2); print(header2)

    # Save text summary
    text_results_filename = os.path.join(output_dir, f"evaluation_summary_forced_{timestamp}.txt")
    try:
        with open(text_results_filename, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Run Timestamp: {timestamp}\n")
            f.write(f"Evaluation Mode: Lane Changes ONLY when Proposed AND Agreed by Agent\n")
            f.write(f"Evaluation Episodes: {eval_config.EVAL_EPISODES}\n")
            f.write(f"Max Steps per Episode: {eval_config.EVAL_MAX_STEPS}\n")
            f.write(f"DQN Model: {os.path.basename(eval_config.DQN_MODEL_PATH)}\n")
            f.write(f"PPO Model: {os.path.basename(eval_config.PPO_MODEL_PATH)}\n")
            f.write(f"SUMO Config: {eval_config.EVAL_SUMO_CONFIG}\n")
            f.write(f"Ego Insertion Delay: {eval_config.EGO_INSERTION_DELAY_SECONDS} seconds\n")
            f.write(f"Force Change Interval: {eval_config.FORCE_CHANGE_INTERVAL_STEPS} steps\n")
            f.write("\n--- Results Comparison ---\n")
            f.write("\n".join(comparison_lines))
        print(f"Text results summary saved to: {text_results_filename}")
    except Exception as e: print(f"Error saving text results: {e}")


    # --- Generate Comparison Plots ---
    print("\n--- Generating Comparison Plots ---")
    models = ['DQN', 'PPO']
    colors = ['deepskyblue', 'lightcoral']

    # --- 1. Bar Plots for Key Metrics ---
    plot_metrics_bar = [ # (key, title, ylabel, filename_suffix, std_dev_key) - Updated keys/titles
        ('avg_steps', '每回合平均步数', '步数', 'avg_steps', 'std_steps'),
        ('avg_speed_episode', '每回合平均速度', '速度 (m/s)', 'avg_speed_per_episode', 'std_speed_episode'),
        ('collision_rate', '总碰撞率', '比率 (%)', 'collision_rate', None),
        ('avg_dist', '平均行驶距离', '距离 (m)', 'avg_dist', 'std_dist'),
        ('avg_agreed_lc', '平均同意换道次数', '次数', 'avg_agreed_lc', None),
        ('forced_agreement_rate', '强制换道: 同意率', '比率 (%)', 'forced_agreement_rate', None),
        ('forced_disagreement_rate', '强制换道: 不同意率', '比率 (%)', 'forced_disagreement_rate', None), # New plot
        ('forced_execution_success_rate_agreed', '同意换道: 执行成功率', '比率 (%)', 'forced_exec_success_rate', None),
        ('forced_execution_collision_rate_agreed', '同意换道: 执行碰撞率', '比率 (%)', 'forced_exec_collision_rate', None),
    ]

    for key, title, ylabel, fname_suffix, std_key in plot_metrics_bar:
        plt.figure(figsize=(7, 6)); # Slightly larger figure
        plot_title = f'{title}\n(DQN vs PPO - Forced Only Mode)'
        plt.title(plot_title, fontsize=14)

        dqn_val = results.get('dqn', {}).get(key)
        ppo_val = results.get('ppo', {}).get(key)

        # Skip plot if data for either model is missing for this key
        if dqn_val is None or ppo_val is None:
            print(f"Skipping plot for '{title}' due to missing data.")
            plt.close()
            continue

        values = [dqn_val, ppo_val]
        std_devs = [0, 0]
        has_std = False
        if std_key:
             std_dqn = results.get('dqn', {}).get(std_key)
             std_ppo = results.get('ppo', {}).get(std_key)
             # Check if std dev is numeric and valid
             if isinstance(std_dqn, (int, float)) and np.isfinite(std_dqn) and \
                isinstance(std_ppo, (int, float)) and np.isfinite(std_ppo):
                std_devs = [std_dqn, std_ppo]
                has_std = True

        # Determine precision from the metric definition
        fmt_str = next((m[2] for m in metrics_definitions if m[0] == key), ".1f")
        precision = int(fmt_str[-2]) if len(fmt_str) >= 2 and fmt_str[-1] == 'f' else 1

        bars = plt.bar(models, values, color=colors, yerr=std_devs if has_std else None, capsize=5 if has_std else 0)

        # Add labels to bars
        for i, bar in enumerate(bars):
             yval = bar.get_height()
             y_offset = yval + (std_devs[i] * 0.6 if has_std else 0) + max(values)*0.02 # Adjusted offset
             # Format label text
             label_text = f"{yval:.{precision}f}"
             if '%' in ylabel: label_text += "%"
             plt.text(bar.get_x() + bar.get_width()/2.0, y_offset, label_text, ha='center', va='bottom', fontsize=10)

        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)

        # Adjust Y limits dynamically
        max_val_display = max(yval + (std_devs[i] if has_std else 0) for i, yval in enumerate(values))
        min_val_display = min(yval - (std_devs[i] if has_std else 0) for i, yval in enumerate(values))
        if '%' in ylabel: plt.ylim(0, max(1, max_val_display * 1.15, 105)) # Ensure 0-100+ range for percentages
        elif max_val_display > 0 : plt.ylim(bottom=max(0, min_val_display * 0.9), top=max_val_display * 1.15) # Allow slightly below zero if std pushes it
        elif max_val_display == 0: plt.ylim(-0.1, 1) # Handle cases where value is zero
        else: plt.ylim(bottom=min_val_display*1.15, top=max(0.1, max_val_display * 0.8)) # Both negative range

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plot_filename = os.path.join(output_dir, f"plot_bar_{fname_suffix}_{timestamp}.png")
        try:
            plt.savefig(plot_filename); print(f"Bar plot saved: {plot_filename}")
        except Exception as e: print(f"Error saving plot {plot_filename}: {e}")
        plt.close()

    # --- 2. Distribution Plots (Box Plots) ---
    print("\n--- Generating Distribution Plots (Box Plots) ---")

    # Speed Distribution (All Steps)
    if dqn_all_speeds or ppo_all_speeds:
        plt.figure(figsize=(7, 5)); plt.title('所有步骤的速度分布 (DQN vs PPO - Forced Only)', fontsize=14)
        speed_data = []
        labels = []
        if dqn_all_speeds: speed_data.append(dqn_all_speeds); labels.append('DQN')
        if ppo_all_speeds: speed_data.append(ppo_all_speeds); labels.append('PPO')
        if speed_data:
            plt.boxplot(speed_data, labels=labels, showfliers=True) # Show outliers
            plt.ylabel('速度 (m/s)'); plt.grid(axis='y', linestyle='--'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"plot_box_speed_distribution_{timestamp}.png")); print("Speed distribution plot saved.")
        else: print("Skipping speed distribution plot (no data).")
        plt.close()
    else: print("Skipping speed distribution plot (no data).")


    # Min Front Distance Distribution (Per Episode)
    dqn_min_dists = [r.min_front_dist for r in dqn_results_list]
    ppo_min_dists = [r.min_front_dist for r in ppo_results_list]
    if dqn_min_dists or ppo_min_dists:
        plt.figure(figsize=(7, 5)); plt.title('每回合最小前车距离分布 (DQN vs PPO - Forced Only)', fontsize=14)
        dist_data = []
        labels = []
        if dqn_min_dists: dist_data.append(dqn_min_dists); labels.append('DQN')
        if ppo_min_dists: dist_data.append(ppo_min_dists); labels.append('PPO')
        if dist_data:
            plt.boxplot(dist_data, labels=labels, showfliers=True)
            plt.ylabel('最小前车距离 (m)'); plt.ylim(bottom=0); plt.grid(axis='y', linestyle='--'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"plot_box_min_front_dist_{timestamp}.png")); print("Min front distance distribution plot saved.")
        else: print("Skipping min front distance plot (no data).")
        plt.close()
    else: print("Skipping min front distance plot (no data).")

    # Add more box plots if needed (e.g., steps, distance per episode) following the same pattern...

    # --- 3. Bar Plots (Lane Occupancy, Action Distribution) ---
    print("\n--- Generating Step Aggregate Plots (Bar Plots) ---")

    # Lane Occupancy
    dqn_lane_data = results.get('dqn', {}).get('step_level_aggregates', {}).get('lane_occupancy_percent')
    ppo_lane_data = results.get('ppo', {}).get('step_level_aggregates', {}).get('lane_occupancy_percent')
    if dqn_lane_data or ppo_lane_data:
        plt.figure(figsize=(8, 5)); plt.title('车道占用率 (DQN vs PPO - Forced Only)', fontsize=14)
        lane_labels = [f'车道 {i}' for i in range(eval_config.NUM_LANES)]
        dqn_lane_perc = [dqn_lane_data.get(f'lane_{i}', 0.0) if dqn_lane_data else 0.0 for i in range(eval_config.NUM_LANES)]
        ppo_lane_perc = [ppo_lane_data.get(f'lane_{i}', 0.0) if ppo_lane_data else 0.0 for i in range(eval_config.NUM_LANES)]
        x = np.arange(len(lane_labels)); width = 0.35
        rects1 = plt.bar(x - width/2, dqn_lane_perc, width, label='DQN', color=colors[0])
        rects2 = plt.bar(x + width/2, ppo_lane_perc, width, label='PPO', color=colors[1])
        plt.ylabel('占用时间 (%)'); plt.xticks(x, lane_labels); plt.ylim(0, 105); plt.legend()
        plt.bar_label(rects1, padding=3, fmt='%.1f%%'); plt.bar_label(rects2, padding=3, fmt='%.1f%%')
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"plot_bar_lane_occupancy_{timestamp}.png")); print("Lane occupancy plot saved.")
        plt.close()
    else: print("Skipping lane occupancy plot (no data).")


    # Action Distribution (Executed Actions)
    dqn_action_data = results.get('dqn', {}).get('step_level_aggregates', {}).get('action_distribution_percent')
    ppo_action_data = results.get('ppo', {}).get('step_level_aggregates', {}).get('action_distribution_percent')
    if dqn_action_data or ppo_action_data:
        plt.figure(figsize=(8, 5)); plt.title('执行动作分布 (DQN vs PPO - Forced Only)', fontsize=14)
        action_labels = ['保持车道', '向左变道 (同意)', '向右变道 (同意)'] # Updated labels
        dqn_action_perc = [dqn_action_data.get(f'action_{i}', 0.0) if dqn_action_data else 0.0 for i in range(3)]
        ppo_action_perc = [ppo_action_data.get(f'action_{i}', 0.0) if ppo_action_data else 0.0 for i in range(3)]
        x = np.arange(len(action_labels)); width = 0.35
        rects1 = plt.bar(x - width/2, dqn_action_perc, width, label='DQN', color=colors[0])
        rects2 = plt.bar(x + width/2, ppo_action_perc, width, label='PPO', color=colors[1])
        plt.ylabel('选择频率 (%)'); plt.xticks(x, action_labels); plt.ylim(0, 105); plt.legend()
        plt.bar_label(rects1, padding=3, fmt='%.1f%%'); plt.bar_label(rects2, padding=3, fmt='%.1f%%')
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"plot_bar_action_distribution_{timestamp}.png")); print("Action distribution plot saved.")
        plt.close()
    else: print("Skipping action distribution plot (no data).")

    # --- Save JSON Data ---
    print("\n--- Saving Detailed Data ---")
    data_filename = os.path.join(output_dir, f"evaluation_data_forced_{timestamp}.json")
    # Add raw episode results and config to the results dict
    results['eval_config'] = {k: v for k, v in vars(EvalConfig).items() if not k.startswith('__')}
    if 'dqn' in results and dqn_results_list:
        results['dqn']['raw_results'] = [r._asdict() for r in dqn_results_list]
    if 'ppo' in results and ppo_results_list:
        results['ppo']['raw_results'] = [r._asdict() for r in ppo_results_list]
    # We already added step_level_aggregates earlier

    try:
        # Enhanced JSON Encoder for Numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating):
                     # Handle NaN/Inf specifically for JSON compatibility
                     if not np.isfinite(obj): return str(obj) # Store as "NaN" or "Infinity"
                     return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist() # Convert arrays to lists
                if isinstance(obj, np.bool_): return bool(obj) # Convert numpy bool to python bool
                return super(NpEncoder, self).default(obj)

        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        print(f"Detailed evaluation data saved to: {data_filename}")
    except Exception as e: print(f"Error saving evaluation data JSON: {e}")

if __name__ == "__main__":
    main()