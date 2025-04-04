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
from torch.distributions import Categorical
import traci
from tqdm import tqdm
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import ReduceLROnPlateau # Replaced with linear decay
from typing import List, Tuple, Dict, Optional, Any
import socket # 用于端口检查
import traceback # 用于打印详细错误
import collections # For deque in normalization

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

#####################
#     配置区域       #
#####################
class Config:
    # --- SUMO 配置 ---
    sumo_binary = "sumo" # 或 "sumo-gui"
    config_path = "a.sumocfg"
    step_length = 0.2
    ego_vehicle_id = "drl_ego_car"
    ego_type_id = "car_ego"
    port_range = (8890, 8900)

    # --- 行为克隆 (BC) ---
    use_bc = True
    bc_collect_episodes = 15 # Increased slightly
    bc_epochs = 20           # Increased slightly
    bc_learning_rate = 1e-4

    # --- PPO 训练 ---
    ppo_episodes = 500 # Increased significantly
    max_steps = 8000 # (1600s simulation time limit per episode)
    log_interval = 10
    save_interval = 50

    # --- PPO 超参数 ---
    gamma = 0.99
    clip_epsilon = 0.2
    initial_learning_rate = 3e-4 # Renamed for clarity with decay
    final_learning_rate = 1e-6   # Target LR for linear decay
    batch_size = 512
    ppo_epochs = 5
    hidden_size = 256 # Start with 256, consider 512 if needed
    gae_lambda = 0.95
    value_clip = True
    value_clip_epsilon = 0.2
    normalize_advantages = True
    gradient_clip_norm = 1.0

    # --- Normalization ---
    normalize_observations = True # Enable observation normalization
    normalize_rewards = True      # Enable reward normalization (scaling)
    obs_norm_clip = 5.0           # Clip normalized observations to [-5, 5]
    reward_norm_clip = 10.0         # Clip normalized rewards to [-10, 10]
    norm_update_rate = 0.001      # Rate for updating running mean/std (ema alpha)

    # --- 熵正则化 (鼓励探索) ---
    use_entropy_decay = True
    entropy_coef_start = 0.05
    entropy_coef_end = 0.005
    # Decay over most of the extended training period
    entropy_decay_episodes = int(ppo_episodes * 0.8)

    # --- 状态/动作空间 ---
    # State: [ego_speed, lane_idx,
    #         front_dist, front_rel_speed,
    #         lf_dist, lf_rel_speed, lb_dist,
    #         rf_dist, rf_rel_speed, rb_dist,
    #         can_left, can_right] (Indices 0-11)
    state_dim = 12
    action_dim = 3 # 0: Keep, 1: Left, 2: Right

    # --- 环境参数 ---
    max_speed_global = 33.33 # m/s (~120 km/h)
    max_distance = 100.0     # m
    lane_max_speeds = [33.33, 27.78, 22.22] # m/s - Must match a.net.xml

    # --- Reward Function Parameters (REVISED) ---
    reward_collision = -100.0
    reward_high_speed_scale = 0.15          # Slightly increased emphasis on speed
    reward_low_speed_penalty_scale = 0.1    # Slightly increased penalty for low speed
    reward_lane_change_penalty = -0.1
    # reward_progress_scale = 0.01          # REMOVED
    time_alive_reward = 0.01                # ADDED: Small reward per step for survival
    reward_comfort_penalty_scale = 0.05     # ADDED: Penalty for harsh braking
    target_speed_factor = 0.95
    safe_distance_penalty_scale = 0.2       # INCREASED: Stronger penalty for tailgating
    min_buffer_dist_reward = 5.0            # m
    time_gap_reward = 0.8                   # s
    min_buffer_dist_bc = 6.0                # m (BC specific)
    time_gap_bc = 1.8                       # s (BC specific)

#####################
#   Normalization    #
#####################
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
        # Simple count update (less critical with EMA)
        self.count += batch_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

# Reward scaling helper
class RewardNormalizer:
    def __init__(self, gamma: float, epsilon: float = 1e-8, alpha: float = 0.001):
        self.returns = collections.deque(maxlen=1000) # Store recent returns
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.epsilon = epsilon
        self.alpha = alpha

    def update(self, rewards: np.ndarray):
        # Assuming rewards is a list/array of rewards from one trajectory/batch
        # We need to estimate the standard deviation of discounted returns
        # A simpler approximation is to track the variance of recent undiscounted returns
        self.returns.extend(rewards)
        if len(self.returns) > 1:
             current_mean = np.mean(self.returns)
             current_var = np.var(self.returns)
             # EMA update
             self.ret_mean = (1.0 - self.alpha) * self.ret_mean + self.alpha * current_mean
             self.ret_var = (1.0 - self.alpha) * self.ret_var + self.alpha * current_var

    def normalize(self, r: np.ndarray, clip: float = 10.0) -> np.ndarray:
        std = np.sqrt(self.ret_var + self.epsilon)
        norm_r = r / std
        return np.clip(norm_r, -clip, clip)


#####################
#   工具函数         #
#####################
# get_available_port, kill_sumo_processes remain the same
def get_available_port(start_port, end_port):
    """查找指定范围内的可用端口"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError(f"在范围 [{start_port}, {end_port}] 内未找到可用端口。")

def kill_sumo_processes():
    """杀死可能残留的 SUMO 进程"""
    print("正在尝试终止残留的 SUMO 进程...")
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
        if killed: print("已终止一个或多个 SUMO 进程。")
        # else: print("未发现需要终止的 SUMO 进程。") # Less verbose
    except Exception as e: print(f"终止 SUMO 进程时出错: {e}")
    time.sleep(0.5)

def linear_decay(start_val, end_val, total_steps, current_step):
    """线性衰减计算 (can be used for LR or entropy)"""
    if current_step >= total_steps:
        return end_val
    return start_val + (end_val - start_val) * (current_step / total_steps)


#####################
#   SUMO 环境封装    #
#####################
class SumoEnv:
    def __init__(self, config: Config):
        self.config = config
        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_speed = 0.0 # Used for comfort penalty
        self.last_raw_state = np.zeros(config.state_dim) # Store raw state before norm

        # Metrics
        self.reset_metrics()

    def reset_metrics(self):
        """重置回合内的指标"""
        self.change_lane_count = 0
        self.collision_occurred = False
        self.current_step = 0
        self.last_action = 0 # Last executed action

    # _start_sumo remains largely the same
    def _start_sumo(self):
        """启动 SUMO 实例并连接 TraCI"""
        kill_sumo_processes() # Ensure no old processes

        try:
            self.traci_port = get_available_port(self.config.port_range[0], self.config.port_range[1])
            # print(f"Found available port: {self.traci_port}") # Less verbose
        except IOError as e:
             print(f"ERROR: Failed to find available port: {e}")
             sys.exit(1)

        sumo_cmd = [
            self.config.sumo_binary, "-c", self.config.config_path,
            "--remote-port", str(self.traci_port),
            "--step-length", str(self.config.step_length),
            "--collision.check-junctions", "true",
            "--collision.action", "warn", # Let Python handle collision consequences
            "--time-to-teleport", "-1",
            "--no-warnings", "true", # Suppress SUMO warnings
            "--seed", str(np.random.randint(0, 10000))
            # "--log", f"sumo_log_{self.traci_port}.txt" # Optional
        ]
        # print(f"Starting SUMO (Port: {self.traci_port}): {' '.join(sumo_cmd)}") # Less verbose

        try:
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=None, stderr=None)
        except FileNotFoundError:
             print(f"ERROR: SUMO executable '{self.config.sumo_binary}' not found.")
             sys.exit(1)
        except Exception as e:
             print(f"ERROR: Failed to start SUMO process: {e}")
             sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                # print(f"Attempting TraCI connection ({attempt + 1}/{connection_attempts})...") # Less verbose
                time.sleep(1.5 + attempt) # Shorter initial wait
                traci.init(self.traci_port)
                print(f"✅ SUMO TraCI connected (Port: {self.traci_port}).")
                return
            except traci.exceptions.TraCIException as e:
                # print(f"TraCI connection failed: {e}") # Less verbose
                if attempt == connection_attempts - 1:
                    print("Max TraCI connection attempts reached.")
                    self._close()
                    raise ConnectionError(f"Could not connect to SUMO (Port: {self.traci_port}).")
            except Exception as e:
                print(f"Unexpected error connecting TraCI: {e}")
                self._close()
                raise ConnectionError(f"Unknown error connecting to SUMO (Port: {self.traci_port}).")

    # _add_ego_vehicle remains largely the same
    def _add_ego_vehicle(self):
        """向模拟中添加 Ego 车辆"""
        ego_route_id = "route_E0"
        if ego_route_id not in traci.route.getIDList():
             print(f"Warning: Route '{ego_route_id}' not found. Adding route based on edge 'E0'.")
             try: traci.route.add(ego_route_id, ["E0"])
             except traci.exceptions.TraCIException as e: raise RuntimeError(f"Failed to add route '{ego_route_id}': {e}")

        if self.config.ego_type_id not in traci.vehicletype.getIDList():
            # print(f"Ego type '{self.config.ego_type_id}' not found, copying from 'car'.") # Less verbose
            try:
                traci.vehicletype.copy("car", self.config.ego_type_id)
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0")
                # print(f"Parameters set for Ego type '{self.config.ego_type_id}'.") # Less verbose
            except traci.exceptions.TraCIException as e: print(f"Warning: Failed to set parameters for Ego type '{self.config.ego_type_id}': {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            # print(f"Warning: Residual Ego vehicle '{self.config.ego_vehicle_id}' detected, removing.") # Less verbose
            try:
                traci.vehicle.remove(self.config.ego_vehicle_id)
                time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"Warning: Failed to remove residual Ego: {e}")

        try:
            # print(f"Adding Ego vehicle '{self.config.ego_vehicle_id}'...") # Less verbose
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                              typeID=self.config.ego_type_id, depart="now",
                              departLane="random", departSpeed="max")

            wait_steps = int(2.0 / self.config.step_length)
            ego_appeared = False
            for i in range(wait_steps):
                traci.simulationStep()
                if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                     # print(f"✅ Ego '{self.config.ego_vehicle_id}' added successfully (Step: {i+1}).") # Less verbose
                     ego_appeared = True
                     break
            if not ego_appeared:
                raise RuntimeError(f"Ego vehicle did not appear within {wait_steps} steps.")

        except traci.exceptions.TraCIException as e:
            print(f"ERROR: Failed to add Ego vehicle '{self.config.ego_vehicle_id}': {e}")
            print(f"Available routes: {traci.route.getIDList()}")
            print(f"Available vTypes: {traci.vehicletype.getIDList()}")
            raise RuntimeError("Failed adding Ego vehicle.")


    def reset(self) -> np.ndarray:
        """重置环境，开始新回合"""
        # print("\n" + "="*10 + " Resetting Environment " + "="*10) # Less verbose
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.reset_metrics()

        # Initialize last_speed and last_raw_state
        self.last_speed = 0.0
        self.last_raw_state = np.zeros(self.config.state_dim)
        initial_state = np.zeros(self.config.state_dim) # Default to zero state
        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
             try:
                 self.last_speed = traci.vehicle.getSpeed(self.config.ego_vehicle_id)
                 initial_state = self._get_state() # Get actual initial state
                 self.last_raw_state = initial_state.copy() # Store the raw state
             except traci.exceptions.TraCIException:
                 pass # Keep defaults if error

        # print("Environment reset complete.") # Less verbose
        # print(f"Initial raw state: {np.round(self.last_raw_state, 2)}")
        return self.last_raw_state # Return the raw state

    # _get_surrounding_vehicle_info remains the same
    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """获取关键区域最近车辆的距离和相对速度"""
        max_dist = self.config.max_distance
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                dx = veh_pos[0] - ego_pos[0] # Longitudinal distance on straight road
                distance = abs(dx)
                if distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed # ego faster -> positive

                if veh_lane == ego_lane:
                    if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                elif veh_lane == ego_lane - 1: # Left lane
                    if dx > 0 and distance < infos['left_front'][0]: infos['left_front'] = (distance, rel_speed)
                    elif dx <= 0 and distance < infos['left_back'][0]: infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1: # Right lane
                    if dx > 0 and distance < infos['right_front'][0]: infos['right_front'] = (distance, rel_speed)
                    elif dx <= 0 and distance < infos['right_back'][0]: infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue # Vehicle might have left
        return infos

    def _get_state(self) -> np.ndarray:
        """获取当前环境状态 (raw values, before normalization)"""
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            return state # Return zero state if ego doesn't exist

        try:
            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            # couldChangeLane returns True/False, 0 = cannot, >0 = possible
            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (self.config.action_dim - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # Fill state vector with RAW values (normalization happens in Agent)
            state[0] = ego_speed
            state[1] = float(ego_lane)
            state[2] = surround_info['front'][0]
            state[3] = surround_info['front'][1] # Relative speed
            state[4] = surround_info['left_front'][0]
            state[5] = surround_info['left_front'][1]
            state[6] = surround_info['left_back'][0]
            state[7] = surround_info['right_front'][0]
            state[8] = surround_info['right_front'][1]
            state[9] = surround_info['right_back'][0]
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            # Store this raw state
            self.last_raw_state = state.copy()

        except traci.exceptions.TraCIException as e:
            print(f"Warning: TraCI error getting state for {ego_id}: {e}. Returning last known state.")
            return self.last_raw_state # Return last known good state
        except Exception as e:
            print(f"Warning: Unknown error getting state for {ego_id}: {e}. Returning last known state.")
            traceback.print_exc()
            return self.last_raw_state

        return state


    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行一个动作，返回 (下一原始状态, 奖励, 是否结束)"""
        done = False
        reward = 0.0
        ego_id = self.config.ego_vehicle_id
        self.last_action = action

        if ego_id not in traci.vehicle.getIDList():
             print(f"Warning: Ego '{ego_id}' missing at start of step {self.current_step}.")
             return self.last_raw_state, self.config.reward_collision, True # Return last known state

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id)
            # speed_before_step = traci.vehicle.getSpeed(ego_id) # Use self.last_speed instead

            # 1. Execute Action
            lane_change_initiated = False
            if action == 1 and current_lane > 0:
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
                self.change_lane_count += 1
                lane_change_initiated = True
            elif action == 2 and current_lane < (self.config.action_dim - 1):
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
                self.change_lane_count += 1
                lane_change_initiated = True

            # 2. Simulation Step
            traci.simulationStep()
            self.current_step += 1

            # 3. Check Status & Calculate Reward
            # Check if ego still exists
            if ego_id not in traci.vehicle.getIDList():
                # print(f"Ego '{ego_id}' disappeared after step {self.current_step}. Assuming collision.") # Less verbose
                self.collision_occurred = True
                done = True
                reward = self.config.reward_collision
                # Return last known raw state before disappearance
                next_state = self.last_raw_state
                return next_state, reward, done

            # Ego exists, check collision list
            collisions = traci.simulation.getCollisions()
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    # print(f"💥 Collision DETECTED! Ego: {ego_id}, Other: {col.victim if col.collider==ego_id else col.collider}") # Less verbose
                    self.collision_occurred = True
                    done = True
                    reward = self.config.reward_collision
                    break

            # Calculate reward if no collision detected yet
            next_state_raw = self._get_state() # Get the raw state after the step
            if not self.collision_occurred:
                # Pass necessary info from the NEW state to reward function
                current_speed_after_step = next_state_raw[0]
                current_lane_after_step = int(next_state_raw[1])
                front_dist_after_step = next_state_raw[2]
                reward = self._calculate_reward(action, current_speed_after_step,
                                                current_lane_after_step, front_dist_after_step)

            # 4. Check other termination conditions
            if traci.simulation.getTime() >= 3600: done = True
            if self.current_step >= self.config.max_steps: done = True

        except traci.exceptions.TraCIException as e:
            print(f"ERROR: TraCI exception during step {self.current_step}: {e}")
            if ego_id not in traci.vehicle.getIDList():
                 self.collision_occurred = True
                 reward = self.config.reward_collision
            done = True
            next_state_raw = self.last_raw_state # Use last known state on error
        except Exception as e:
            print(f"ERROR: Unknown exception during step {self.current_step}: {e}")
            traceback.print_exc()
            done = True
            reward = self.config.reward_collision
            self.collision_occurred = True
            next_state_raw = self.last_raw_state

        # 5. Update last_speed for next step's reward calculation
        # Use the speed from the state we just calculated
        self.last_speed = next_state_raw[0]

        # Return the RAW next state
        return next_state_raw, reward, done


    # REVISED REWARD CALCULATION
    def _calculate_reward(self, action: int, current_speed: float, current_lane: int, front_dist: float) -> float:
        """计算当前状态的奖励 (基于步进后的状态信息)"""
        ego_id = self.config.ego_vehicle_id

        # Should not happen if called correctly, but double-check
        if self.collision_occurred: return 0.0

        try:
            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            # 1. Speed Reward / Penalty
            speed_diff = abs(current_speed - target_speed)
            # Exponential reward for being close to target
            speed_reward = np.exp(- (speed_diff / (target_speed + 1e-6))**2 ) * self.config.reward_high_speed_scale
            # Penalty for being too slow
            low_speed_penalty = 0.0
            low_speed_threshold = target_speed * 0.6
            if current_speed < low_speed_threshold:
                # Linear penalty: -scale * (1 - speed/threshold)
                low_speed_penalty = (current_speed / low_speed_threshold - 1.0) * self.config.reward_low_speed_penalty_scale # Negative value

            # 2. Lane Change Penalty
            lane_change_penalty = self.config.reward_lane_change_penalty if action != 0 else 0.0

            # 3. Safety Distance Penalty (using front_dist passed from step)
            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < min_safe_dist and front_dist > 0: # Ensure distance > 0
                 # Stronger penalty: scale * ( (safe_dist/actual_dist)^2 - 1 ) might be too aggressive
                 # Linear penalty (increased scale):
                 safety_dist_penalty = max(-1.0, (front_dist / min_safe_dist - 1.0)) * self.config.safe_distance_penalty_scale # Negative value, capped

            # 4. Comfort Penalty (Harsh Braking)
            comfort_penalty = 0.0
            speed_change = current_speed - self.last_speed # Speed change during this step
            braking_threshold = -2.0 # m/s^2 equivalent (approx -2.0 m/s per 0.2s step is -10 m/s^2)
            # We approximate accel = speed_change / step_length
            approx_accel = speed_change / self.config.step_length
            if approx_accel < braking_threshold:
                # Penalize based on how much braking exceeds threshold
                comfort_penalty = (approx_accel - braking_threshold) * self.config.reward_comfort_penalty_scale # Negative value

            # 5. Time Alive Reward
            time_alive = self.config.time_alive_reward

            # Total Reward
            total_reward = (speed_reward +
                            low_speed_penalty +
                            lane_change_penalty +
                            safety_dist_penalty +
                            comfort_penalty +
                            time_alive)

            # Debug print (optional, can be noisy)
            # print(f"R={total_reward:.3f} [Spd={speed_reward:.3f}|{low_speed_penalty:.3f} LC={lane_change_penalty:.3f} Safe={safety_dist_penalty:.3f} Comf={comfort_penalty:.3f} Alive={time_alive:.3f}]")

            return total_reward

        except Exception as e:
            print(f"Warning: Error calculating reward: {e}. Returning 0.")
            traceback.print_exc()
            return 0.0

    # _close remains the same
    def _close(self):
        """关闭 SUMO 实例和 TraCI 连接"""
        if self.sumo_process:
            # print("Closing SUMO connection...") # Less verbose
            try:
                traci.close()
            except (traci.exceptions.TraCIException, ConnectionResetError, Exception):
                 pass # Ignore errors on close, process will be killed
            finally:
                # print("Terminating SUMO process...") # Less verbose
                try:
                    if self.sumo_process.poll() is None: # Check if still running
                        self.sumo_process.terminate()
                        self.sumo_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # print("SUMO did not terminate gracefully, killing.") # Less verbose
                    self.sumo_process.kill()
                    self.sumo_process.wait(timeout=2) # Wait after kill
                except Exception as e:
                    print(f"Warning: Error during SUMO termination: {e}")
                self.sumo_process = None
                self.traci_port = None
                # print("SUMO resources cleaned.") # Less verbose
                time.sleep(0.5)
        else:
            self.traci_port = None

#####################
#   规则策略 (BC用)  #
#####################
# rule_based_action_improved remains the same
def rule_based_action_improved(state: np.ndarray, config: Config) -> int:
    """改进的基于规则的策略，用于 BC 数据收集 (更保守)"""
    # State indices (assuming state is RAW, not normalized here for rule logic)
    EGO_SPEED, LANE_IDX, FRONT_DIST, FRONT_REL_SPEED, \
    LF_DIST, LF_REL_SPEED, LB_DIST, \
    RF_DIST, RF_REL_SPEED, RB_DIST, \
    CAN_LEFT, CAN_RIGHT = range(config.state_dim)

    # Get raw values from state
    ego_speed = state[EGO_SPEED]
    lane_idx = int(state[LANE_IDX])
    front_dist = state[FRONT_DIST]
    front_rel_speed = state[FRONT_REL_SPEED] # ego - front
    lf_dist = state[LF_DIST]
    lb_dist = state[LB_DIST]
    rf_dist = state[RF_DIST]
    rb_dist = state[RB_DIST]
    can_change_left = state[CAN_LEFT] > 0.5
    can_change_right = state[CAN_RIGHT] > 0.5

    # Lane speeds
    current_max_speed = config.lane_max_speeds[lane_idx]
    left_max_speed = config.lane_max_speeds[lane_idx - 1] if lane_idx > 0 else -1
    right_max_speed = config.lane_max_speeds[lane_idx + 1] if lane_idx < (config.action_dim - 1) else -1

    # BC Safety parameters (more conservative)
    reaction_time_gap = config.time_gap_bc
    min_buffer_dist = config.min_buffer_dist_bc
    safe_follow_dist = min_buffer_dist + ego_speed * reaction_time_gap
    required_front_gap = min_buffer_dist + ego_speed * (reaction_time_gap * 0.9) # For lane change
    required_back_gap = min_buffer_dist + ego_speed * (reaction_time_gap * 0.6)  # For lane change

    # 1. Emergency Avoidance
    emergency_dist = min_buffer_dist + ego_speed * (reaction_time_gap * 0.4)
    if front_dist < emergency_dist and front_rel_speed > 2.0:
        left_safe = can_change_left and (lf_dist > required_front_gap * 0.8) and (lb_dist > required_back_gap * 0.8)
        right_safe = can_change_right and (rf_dist > required_front_gap * 0.8) and (rb_dist > required_back_gap * 0.8)
        if left_safe and right_safe: return 1 # Prioritize left slightly in emergency?
        if left_safe: return 1
        if right_safe: return 2
        return 0 # Brake hard

    # 2. Normal Avoidance
    if front_dist < safe_follow_dist and front_rel_speed > 1.0:
        left_safe = can_change_left and (lf_dist > required_front_gap) and (lb_dist > required_back_gap)
        right_safe = can_change_right and (rf_dist > required_front_gap) and (rb_dist > required_back_gap)
        prefer_left = left_safe and (left_max_speed > right_max_speed or not right_safe)
        prefer_right = right_safe and (right_max_speed > left_max_speed or not left_safe)
        if prefer_left: return 1
        if prefer_right: return 2
        if left_safe: return 1 # If speeds equal, prefer left if safe
        if right_safe: return 2 # Else prefer right if safe
        return 0

    # 3. Speed Seeking (Low priority, conservative)
    speed_threshold = current_max_speed * 0.8
    if ego_speed < speed_threshold:
        if can_change_left and left_max_speed > current_max_speed * 1.1: # Left is significantly faster
            left_extra_safe = (lf_dist > required_front_gap * 1.8) and (lb_dist > required_back_gap * 1.5)
            if left_extra_safe: return 1
        if can_change_right and right_max_speed > current_max_speed * 1.1: # Right is significantly faster
            right_extra_safe = (rf_dist > required_front_gap * 1.8) and (rb_dist > required_back_gap * 1.5)
            if right_extra_safe: return 2

    # 4. Default: Keep Lane
    return 0

#####################
#   BC Actor 网络    #
#####################
# BehaviorCloningNet remains the same
class BehaviorCloningNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(BehaviorCloningNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim) # Output raw logits
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

# bc_train remains largely the same, but uses normalized states if enabled
def bc_train(config: Config, bc_data: List[Tuple[np.ndarray, int]], obs_normalizer: Optional[RunningMeanStd]) -> Optional[BehaviorCloningNet]:
    """训练行为克隆网络"""
    if not config.use_bc or not bc_data:
        print("Skipping BC training.")
        return None

    print(f"\n--- Starting BC Training ({len(bc_data)} samples) ---")
    net = BehaviorCloningNet(config.state_dim, config.action_dim, config.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=config.bc_learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    try:
        states_raw = np.array([d[0] for d in bc_data])
        actions = torch.LongTensor(np.array([d[1] for d in bc_data]))

        # Normalize states if enabled
        if config.normalize_observations and obs_normalizer:
             # Update normalizer with BC data first
             obs_normalizer.update(states_raw)
             states_normalized = (states_raw - obs_normalizer.mean) / (obs_normalizer.std + 1e-8)
             states_normalized = np.clip(states_normalized, -config.obs_norm_clip, config.obs_norm_clip)
             states_tensor = torch.FloatTensor(states_normalized)
             print("BC data normalized using running stats.")
        else:
             states_tensor = torch.FloatTensor(states_raw) # Use raw states

    except Exception as e:
        print(f"ERROR preparing BC data: {e}")
        return None

    dataset = torch.utils.data.TensorDataset(states_tensor, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    net.train()
    all_bc_losses = []
    for epoch in range(config.bc_epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            logits = net(batch_states)
            loss = loss_fn(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        all_bc_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == config.bc_epochs - 1:
            print(f"[BC] Epoch {epoch+1}/{config.bc_epochs}, Avg Loss = {avg_loss:.6f}")

    net.eval()
    print("BC training finished.")

    # Plot BC loss curve
    plt.figure("BC Loss Curve", figsize=(8, 4))
    plt.plot(all_bc_losses)
    plt.title("BC Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)

    return net


#####################
#   PPO 网络        #
#####################
# PPO class remains the same
class PPO(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    def get_value(self, x: torch.Tensor) -> torch.Tensor: return self.critic(x)
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor: return self.actor(x)


#####################
#    PPO Agent      #
#####################
class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.policy = PPO(config.state_dim, config.action_dim, config.hidden_size)
        # Use initial learning rate for optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.initial_learning_rate, eps=1e-5)
        # LR scheduler removed, will implement linear decay manually

        # Normalization (initialized here)
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate) if config.normalize_rewards else None

        # Memory: Stores (raw_state, action, log_prob, reward, done, next_raw_state)
        self.memory: List[Tuple[np.ndarray, int, float, float, bool, np.ndarray]] = []
        # Training metrics
        self.training_metrics: Dict[str, List[float]] = {
             "actor_losses": [], "critic_losses": [], "total_losses": [], "entropies": []
        }

    # load_bc_actor remains the same
    def load_bc_actor(self, bc_net: Optional[BehaviorCloningNet]):
        """从 BC 网络加载 Actor 的权重 (如果 BC 被执行)"""
        if bc_net is None:
            print("BC net not available, skipping weight loading.")
            return
        try:
            self.policy.actor.load_state_dict(bc_net.actor.state_dict())
            print("✅ BC Actor weights loaded into PPO policy.")
        except Exception as e:
            print(f"ERROR loading BC Actor weights: {e}")

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running mean/std."""
        if self.config.normalize_observations and self.obs_normalizer:
            # Do not update normalizer here, only during training update
            norm_state = (state - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8)
            return np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip)
        return state # Return raw state if normalization is off

    def get_action(self, raw_state: np.ndarray, current_episode: int) -> Tuple[int, float]:
        """根据当前状态和策略选择动作"""
        if not isinstance(raw_state, np.ndarray) or raw_state.shape != (self.config.state_dim,):
             print(f"Warning: get_action received invalid state type/shape: {type(raw_state)}, {raw_state.shape}. Using zeros.")
             raw_state = np.zeros(self.config.state_dim) # Fallback

        # Normalize the state for policy input
        normalized_state = self.normalize_state(raw_state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)

        self.policy.eval()
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
        self.policy.train()

        # Action masking (using RAW state for lane index)
        # raw_state[1] is the actual lane index
        lane_idx = int(raw_state[1])
        mask = torch.ones_like(probs, dtype=torch.float32)
        if lane_idx == 0: mask[0, 1] = 0.0 # No left turn from lane 0
        elif lane_idx == (self.config.action_dim - 1): mask[0, 2] = 0.0 # No right turn from last lane

        masked_probs = probs * mask
        probs_sum = masked_probs.sum(dim=-1, keepdim=True)

        # Handle cases where all actions might be masked (should be rare)
        if probs_sum.item() < 1e-8:
            valid_indices = torch.where(mask[0] > 0.5)[0]
            if len(valid_indices) == 0: action_idx = 0 # Default to keep lane
            else: action_idx = np.random.choice(valid_indices.numpy()) # Random among valid
            # Create fallback uniform probability for log_prob calculation
            final_probs = torch.zeros_like(probs)
            if len(valid_indices)>0: final_probs[0, valid_indices] = 1.0 / len(valid_indices)
            else: final_probs[0,0] = 1.0 # If even keep lane invalid? Fallback needed
        else:
            final_probs = masked_probs / probs_sum # Renormalize

        # Add small epsilon for numerical stability before Categorical
        final_probs = final_probs + 1e-9
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)

        # Sample action
        try:
             dist = Categorical(probs=final_probs)
             action = dist.sample()
             log_prob = dist.log_prob(action)
             return action.item(), log_prob.item()
        except ValueError as e:
             print(f"ERROR creating Categorical distribution: {e}. Probs: {final_probs}")
             # Fallback to deterministic 'keep lane'
             return 0, torch.log(final_probs[0, 0]).item() # Log prob of keeping lane

    def store(self, transition: Tuple[np.ndarray, int, float, float, bool, np.ndarray]):
        """将经验存入内存 (stores RAW states)"""
        self.memory.append(transition)

    def update(self, current_episode: int, total_episodes: int):
        """使用内存中的数据更新 PPO 策略"""
        if not self.memory: return 0.0

        # --- 1. Data Preparation (Using RAW states from memory) ---
        raw_states = np.array([m[0] for m in self.memory])
        actions = torch.LongTensor(np.array([m[1] for m in self.memory]))
        old_log_probs = torch.FloatTensor(np.array([m[2] for m in self.memory]))
        rewards = np.array([m[3] for m in self.memory])
        dones = torch.BoolTensor(np.array([m[4] for m in self.memory]))
        raw_next_states = np.array([m[5] for m in self.memory])

        # --- Update and Apply Normalization ---
        if self.config.normalize_observations and self.obs_normalizer:
            self.obs_normalizer.update(raw_states) # Update running stats
            # Normalize states used for training
            states_norm = (raw_states - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8)
            next_states_norm = (raw_next_states - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8)
            states_norm = np.clip(states_norm, -self.config.obs_norm_clip, self.config.obs_norm_clip)
            next_states_norm = np.clip(next_states_norm, -self.config.obs_norm_clip, self.config.obs_norm_clip)
            states = torch.FloatTensor(states_norm)
            next_states = torch.FloatTensor(next_states_norm)
        else:
            # Use raw states if normalization is off
            states = torch.FloatTensor(raw_states)
            next_states = torch.FloatTensor(raw_next_states)

        # Normalize rewards if enabled
        if self.config.normalize_rewards and self.reward_normalizer:
             self.reward_normalizer.update(rewards) # Update reward stats
             rewards_normalized = self.reward_normalizer.normalize(rewards, clip=self.config.reward_norm_clip)
             rewards_tensor = torch.FloatTensor(rewards_normalized)
        else:
             rewards_tensor = torch.FloatTensor(rewards)


        # --- 2. Calculate GAE and Returns ---
        with torch.no_grad():
            values = self.policy.get_value(states).squeeze()
            next_values = self.policy.get_value(next_states).squeeze()

        returns = torch.zeros_like(rewards_tensor)
        advantages = torch.zeros_like(rewards_tensor)
        last_gae_lam = 0.0
        num_steps = len(rewards_tensor)

        for t in reversed(range(num_steps)):
            if dones[t]:
                delta = rewards_tensor[t] - values[t]
                last_gae_lam = 0.0
            else:
                next_val = next_values if num_steps == 1 else next_values[t] # Handle single step case
                delta = rewards_tensor[t] + self.config.gamma * next_val - values[t]
            advantages[t] = last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * last_gae_lam
        returns = advantages + values

        # --- 3. Advantage Normalization ---
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 4. PPO Optimization Loop ---
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        epoch_actor_losses, epoch_critic_losses, epoch_total_losses, epoch_entropies = [], [], [], []

        # --- Manual LR Decay ---
        current_lr = linear_decay(self.config.initial_learning_rate,
                                  self.config.final_learning_rate,
                                  total_episodes, current_episode)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        # print(f"Update {current_episode}/{total_episodes}, LR set to {current_lr:.7f}") # Optional debug print

        self.policy.train()
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                # Extract batch data (using normalized states)
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = values[batch_idx]

                # Get new log probs, values, entropy
                new_probs_dist, current_values = self.policy(batch_states)
                current_values = current_values.squeeze()

                # Action masking (using RAW states from the batch indices)
                batch_raw_states = raw_states[batch_idx]
                batch_lanes = torch.tensor(batch_raw_states[:, 1], dtype=torch.long) # Get raw lane index
                mask = torch.ones_like(new_probs_dist, dtype=torch.float32)
                mask[batch_lanes == 0, 1] = 0.0
                mask[batch_lanes == (self.config.action_dim - 1), 2] = 0.0

                masked_new_probs = new_probs_dist * mask
                probs_sum = masked_new_probs.sum(dim=-1, keepdim=True)
                safe_probs_sum = torch.where(probs_sum < 1e-8, torch.ones_like(probs_sum), probs_sum)
                renormalized_probs = masked_new_probs / safe_probs_sum
                renormalized_probs = renormalized_probs + 1e-9
                renormalized_probs = renormalized_probs / renormalized_probs.sum(dim=-1, keepdim=True)

                dist = Categorical(probs=renormalized_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Calculate PPO losses
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                if self.config.value_clip:
                    values_pred_clipped = batch_old_values + torch.clamp(current_values - batch_old_values, -self.config.value_clip_epsilon, self.config.value_clip_epsilon)
                    vf_loss1 = nn.MSELoss()(current_values, batch_returns)
                    vf_loss2 = nn.MSELoss()(values_pred_clipped, batch_returns)
                    critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2) # Removed mean() here, applied below
                else:
                    critic_loss = 0.5 * nn.MSELoss()(current_values, batch_returns)
                # Ensure critic loss is reduced correctly if MSELoss applied reduction="mean" by default
                critic_loss = critic_loss.mean() if not self.config.value_clip else critic_loss # Apply mean if not max(loss1, loss2)


                # Entropy bonus
                current_entropy_coef = self.config.entropy_coef_start
                if self.config.use_entropy_decay:
                    current_entropy_coef = linear_decay(self.config.entropy_coef_start, self.config.entropy_coef_end, self.config.entropy_decay_episodes, current_episode)
                entropy_loss = -current_entropy_coef * entropy

                # Total loss
                loss = actor_loss + critic_loss + entropy_loss

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.gradient_clip_norm)
                self.optimizer.step()

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_total_losses.append(loss.item())
                epoch_entropies.append(entropy.item())

        # --- 5. Update Training Metrics ---
        self.training_metrics["actor_losses"].append(np.mean(epoch_actor_losses))
        self.training_metrics["critic_losses"].append(np.mean(epoch_critic_losses))
        self.training_metrics["total_losses"].append(np.mean(epoch_total_losses))
        self.training_metrics["entropies"].append(np.mean(epoch_entropies))

        # --- 6. Clear Memory ---
        self.memory.clear()

        # Return average UNNORMALIZED reward for logging/comparison
        return np.mean(np.array([m[3] for m in self.memory])) if self.memory else 0.0 # Return avg original reward


    # step_scheduler removed as LR decay is manual now


#####################
#   主训练流程       #
#####################
def main():
    config = Config()

    # --- Basic Checks (remain mostly the same) ---
    # SUMO path check... (keep as before)
    # Config file check... (keep as before)
    # Results dir check... (keep as before)

    # --- Environment and Agent Initialization ---
    # Need separate envs if BC/PPO run in parallel, but sequentially is fine
    env_main = SumoEnv(config) # One env used for both phases
    agent = Agent(config)      # PPO Agent

    bc_net = None # Initialize BC network

    # --- 1. Behavior Cloning (Optional) ---
    if config.use_bc:
        print("\n" + "#"*20 + " Phase 1: Behavior Cloning " + "#"*20)
        bc_data = []
        print("\n--- Starting BC Data Collection ---")
        for ep in range(config.bc_collect_episodes):
            print(f"BC Data Collection - Episode {ep + 1}/{config.bc_collect_episodes}")
            try:
                state_raw = env_main.reset() # Get raw state
                done = False
                ep_steps = 0
                while not done and ep_steps < config.max_steps:
                    # Rule based action needs RAW state
                    action = rule_based_action_improved(state_raw, config)
                    next_state_raw, _, done = env_main.step(action)
                    if not done:
                        bc_data.append((state_raw.copy(), action)) # Store raw state
                    state_raw = next_state_raw
                    ep_steps += 1
                print(f"BC Ep {ep + 1} finished: {ep_steps} steps. Total data: {len(bc_data)}")
            except (ConnectionError, RuntimeError, traci.exceptions.TraCIException) as e:
                print(f"\nERROR during BC episode {ep + 1}: {e}")
                env_main._close() # Ensure cleanup on error
                time.sleep(1)
            except KeyboardInterrupt:
                 print("\nBC data collection interrupted by user.")
                 env_main._close()
                 return

        print(f"\nBC data collection finished. Total samples: {len(bc_data)}")

        # Train BC network (passing the agent's obs_normalizer to potentially update it)
        if bc_data:
             bc_net = bc_train(config, bc_data, agent.obs_normalizer) # Pass normalizer
             if bc_net:
                 agent.load_bc_actor(bc_net)
             else:
                 print("BC training failed, weights not loaded.")
        else:
             print("No BC data collected, skipping training and weight loading.")
    else:
        print("Behavior Cloning disabled.")


    # --- 2. PPO 训练 ---
    print("\n" + "#"*20 + " Phase 2: PPO Training " + "#"*20)

    # --- Logging and Saving Setup ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"bc_ppo_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

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
    all_rewards = []
    lane_change_counts = []
    collision_counts = []
    total_steps_per_episode = []
    best_avg_reward = -float('inf')

    # --- PPO Main Loop ---
    try:
        for episode in tqdm(range(1, config.ppo_episodes + 1), desc="PPO Training Episodes"):
            state_raw = env_main.reset() # Get initial raw state
            episode_reward = 0.0
            done = False
            step_count = 0

            while not done and step_count < config.max_steps:
                # Get action based on raw state (normalization happens inside get_action)
                action, log_prob = agent.get_action(state_raw, episode)
                next_state_raw, reward, done = env_main.step(action) # Env returns raw state

                # Store experience with RAW states and original reward
                agent.store((state_raw.copy(), action, log_prob, reward, done, next_state_raw.copy()))

                state_raw = next_state_raw # Update raw state for next iteration
                episode_reward += reward
                step_count += 1

            # --- Episode End: PPO Update ---
            # print(f"\nEpisode {episode} ended. Reward: {episode_reward:.2f}, Steps: {step_count}, "
            #       f"LaneChanges: {env_main.change_lane_count}, Collision: {env_main.collision_occurred}") # Less verbose
            # Update happens here, using data stored in memory
            avg_original_reward_in_batch = agent.update(episode, config.ppo_episodes) # Pass total episodes for LR decay

            # --- Log metrics ---
            all_rewards.append(episode_reward)
            lane_change_counts.append(env_main.change_lane_count)
            collision_counts.append(1 if env_main.collision_occurred else 0)
            total_steps_per_episode.append(step_count)

            # --- Learning Rate Scheduling done inside agent.update ---

            # --- Save Best Model ---
            avg_window = 20 # Use a slightly larger window for stability
            if episode >= avg_window:
                current_avg_reward = np.mean(all_rewards[-avg_window:])
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    best_model_path = os.path.join(models_dir, "best_model.pth")
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"\n🎉 New best avg reward ({avg_window}ep): {best_avg_reward:.2f}! Model saved.")

            # --- Periodic Logging ---
            if episode % config.log_interval == 0:
                 avg_reward_log = np.mean(all_rewards[-config.log_interval:])
                 avg_steps_log = np.mean(total_steps_per_episode[-config.log_interval:])
                 collision_rate_log = np.mean(collision_counts[-config.log_interval:]) * 100
                 current_lr = agent.optimizer.param_groups[0]['lr'] # Get current LR
                 last_entropy = agent.training_metrics['entropies'][-1] if agent.training_metrics['entropies'] else 'N/A'
                 print(f"\nEp: {episode}/{config.ppo_episodes} | Avg Reward (last {config.log_interval}): {avg_reward_log:.2f} "
                       f"| Best Avg Reward ({avg_window}ep): {best_avg_reward:.2f} "
                       f"| Avg Steps: {avg_steps_log:.1f} | Coll Rate: {collision_rate_log:.1f}% "
                       f"| LR: {current_lr:.7f} | Entropy: {last_entropy if isinstance(last_entropy, str) else last_entropy:.4f}")

            # --- Periodic Model Saving ---
            if episode % config.save_interval == 0:
                 periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                 torch.save(agent.policy.state_dict(), periodic_model_path)
                 # print(f"Periodic model saved: {periodic_model_path}") # Less verbose

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
         print(f"\nFATAL ERROR during training: {e}")
         traceback.print_exc()
    finally:
        print("Closing final environment...")
        env_main._close()

        # --- Save Final Model and Data ---
        if 'agent' in locals() and agent is not None: # Check if agent exists
            last_model_path = os.path.join(models_dir, "last_model.pth")
            torch.save(agent.policy.state_dict(), last_model_path)
            print(f"Final model saved to: {last_model_path}")

            # --- Plotting (remains the same structure) ---
            print("Generating training plots...")
            plt.figure("Training Curves", figsize=(15, 12))

            # Plot 1: Reward + Rolling Mean
            plt.subplot(3, 2, 1); plt.grid(True, linestyle='--')
            plt.plot(all_rewards, label='Episode Reward', alpha=0.6)
            if len(all_rewards) >= 10:
                rolling_avg = np.convolve(all_rewards, np.ones(10)/10, mode='valid')
                plt.plot(np.arange(9, len(all_rewards)), rolling_avg, label='10-Ep Roll Avg', color='red', linewidth=2)
            plt.title("Episode Reward"); plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()

            # Plot 2: Lane Changes
            plt.subplot(3, 2, 2); plt.grid(True, linestyle='--')
            plt.plot(lane_change_counts); plt.title("Lane Changes per Episode")
            plt.xlabel("Episode"); plt.ylabel("Count")

            # Plot 3: Collisions + Rolling Rate
            plt.subplot(3, 2, 3); plt.grid(True, linestyle='--')
            plt.plot(collision_counts, marker='.', linestyle='None', alpha=0.5, label='Collision (1=Yes)')
            if len(collision_counts) >= 10:
                 rolling_coll_rate = np.convolve(collision_counts, np.ones(10)/10, mode='valid') * 100
                 plt.plot(np.arange(9, len(collision_counts)), rolling_coll_rate, label='10-Ep Roll Coll Rate (%)', color='orange')
            plt.title("Collisions & Rolling Rate"); plt.xlabel("Episode"); plt.ylabel("Collision (0/1) / Rate (%)")
            plt.ylim(-5, 105); plt.legend()

            # Plot 4: PPO Losses
            plt.subplot(3, 2, 4); plt.grid(True, linestyle='--')
            if agent.training_metrics["total_losses"]:
                 # Use log scale for better visibility if losses vary greatly
                 # plt.yscale('log')
                 plt.plot(agent.training_metrics["total_losses"], label='Total Loss', alpha=0.8)
                 plt.plot(agent.training_metrics["actor_losses"], label='Actor Loss', alpha=0.6)
                 plt.plot(agent.training_metrics["critic_losses"], label='Critic Loss', alpha=0.6)
                 plt.title("PPO Losses per Update"); plt.xlabel("Update Step"); plt.ylabel("Loss")
                 plt.legend()
            else: plt.title("PPO Losses (No data)")

            # Plot 5: Steps per Episode
            plt.subplot(3, 2, 5); plt.grid(True, linestyle='--')
            plt.plot(total_steps_per_episode); plt.title("Steps per Episode")
            plt.xlabel("Episode"); plt.ylabel("Steps")

            # Plot 6: Entropy
            plt.subplot(3, 2, 6); plt.grid(True, linestyle='--')
            if agent.training_metrics["entropies"]:
                plt.plot(agent.training_metrics["entropies"])
                plt.title("Policy Entropy per Update"); plt.xlabel("Update Step"); plt.ylabel("Avg Entropy")
            else: plt.title("Policy Entropy (No data)")

            plt.tight_layout()
            plot_path = os.path.join(results_dir, "training_curves_final.png")
            plt.savefig(plot_path)
            # plt.show(block=False) # Keep plots from blocking end of script
            plt.close("all") # Close all figures
            print(f"Training plots saved to: {plot_path}")

            # --- Save Training Data JSON ---
            print("Saving training data...")
            training_data = {
                # Config already saved separately
                "episode_rewards": all_rewards,
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "metrics_per_update": agent.training_metrics # Losses and entropy
            }
            data_path = os.path.join(results_dir, "training_data_final.json")
            try:
                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(training_data, f, indent=4, ensure_ascii=False, default=str)
                print(f"Training data saved to: {data_path}")
            except Exception as e:
                print(f"ERROR saving training data: {e}")
        else:
            print("Agent not initialized, cannot save final model/data.")

        print(f"\nTraining finished. Results saved in: {results_dir}")

if __name__ == "__main__":
    main()