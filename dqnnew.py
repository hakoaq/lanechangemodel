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

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

#####################
#     配置区域       #
#####################
class Config:
    # --- SUMO 配置 --- (保持与 PPO 相同)
    sumo_binary = "sumo" # or "sumo-gui"
    config_path = "a.sumocfg"
    step_length = 0.2
    ego_vehicle_id = "drl_ego_car"
    ego_type_id = "car_ego"
    port_range = (8890, 8900)

    # --- 行为克隆 (BC) ---
    use_bc = False # DQN 禁用

    # --- DQN 训练 ---
    dqn_episodes = 1000 # 与 PPO 保持相同的回合数以便比较 <<-- INCREASED FOR REAL TRAINING
    max_steps = 8000   # 与 PPO 保持相同的每回合最大步数
    log_interval = 10
    save_interval = 50

    # --- DQN 超参数 ---
    gamma = 0.99 # 折扣因子 (与 PPO 相同)
    initial_learning_rate = 7e-5 # C51/Noisy 可能需要更低的学习率 (原为 1e-4) # Keep this LR for now
    batch_size = 512 # 从回放缓冲区采样的批量大小 (与 PPO 相同)
    hidden_size = 256 # 网络隐藏层大小 (与 PPO 相同)
    replay_buffer_size = 100000 # 经验回放缓冲区大小
    target_update_freq = 2500   # 目标网络更新的训练 *步数* 间隔 (原为 1000)
    learning_starts = 7500      # 开始训练前收集的步数 (原为 5000)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # 注：您运行的20回合（约4000步）未达到此阈值，因此代理尚未开始训练。
    # Note: Your 20-episode run (~4000 steps) did not reach this threshold, so the agent hasn't started training yet.
    # INCREASE EPISODES OR DECREASE learning_starts TO SEE LEARNING HAPPEN SOONER.
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    use_double_dqn = True       # 使用 Double DQN 改进 (在 C51 逻辑内应用)

    # --- Epsilon-Greedy Exploration (REMOVED - Using Noisy Nets) ---
    # epsilon_start = 1.0
    # epsilon_end = 0.05
    # epsilon_decay_total_steps = int(dqn_episodes * 300 * 0.7) # Estimate based on episodes * avg_steps

    # --- Noisy Networks ---
    use_noisy_nets = True       # 启用 Noisy Networks 进行探索
    noisy_sigma_init = 0.5     # NoisyLinear 层的初始标准差

    # --- Distributional DQN (C51) ---
    use_distributional = True   # 启用 C51 分布式 DQN
    v_min = -110                # 可能的最小回报值 (根据奖励调整: 约 -100 碰撞 + 一些负的步奖励)
    v_max = 30                  # 可能的最大回报值 (因奖励变化略微增加)
    num_atoms = 51              # 分布支撑中的原子数量

    # --- 归一化 --- (保持与 PPO 相同)
    normalize_observations = True # 启用观测值归一化
    normalize_rewards = True      # 启用奖励归一化 (缩放) - 现在在 N-step 计算 *之前* 应用
    obs_norm_clip = 5.0           # 将归一化的观测值裁剪到 [-5, 5]
    reward_norm_clip = 10.0         # 将归一化的奖励裁剪到 [-10, 10]
    norm_update_rate = 0.001      # 更新运行均值/标准差的速率 (EMA alpha)

    # --- Prioritized Experience Replay (PER) ---
    use_per = True
    per_alpha = 0.6             # 优先级指数 (0=均匀, 1=完全优先)
    per_beta_start = 0.4        # 初始 IS 权重指数 (0=无修正, 1=完全修正)
    per_beta_end = 1.0          # 最终 IS 权重指数
    # 在训练期间退火 beta (如果需要，更好地估计步数)
    # 使用更长的训练周期来更好地估计总步数
    per_beta_annealing_steps = int(dqn_episodes * 300 * 0.8) # 在估计总步数的 80% 内退火 # Updated estimation based on more episodes
    per_epsilon = 1e-5          # 添加到优先级的小值

    # --- N-Step Returns ---
    use_n_step = True
    n_step = 5                  # N-step 回报的步数 (原为 3)

    # --- 状态/动作空间 --- (保持与 PPO 相同)
    state_dim = 12
    action_dim = 3 # 0: 保持, 1: 左转, 2: 右转

    # --- 环境参数 --- (保持与 PPO 相同)
    max_speed_global = 33.33 # m/s (~120 km/h)
    max_distance = 100.0     # m
    lane_max_speeds = [33.33, 27.78, 22.22] # m/s - 必须与 a.net.xml 匹配
    num_train_lanes = len(lane_max_speeds) # Added for consistency

    # --- 奖励函数参数 (修订 - 鼓励换道) ---
    reward_collision = -100.0 # 保持原始碰撞惩罚
    reward_high_speed_scale = 0.25 # 增加 (原为 0.15)
    reward_low_speed_penalty_scale = 0.1
    reward_lane_change_penalty = -0.02 # 进一步减少惩罚 (原为 -0.05)
    reward_faster_lane_bonus = 0.6 # *** 新增: 增加移动到更快潜在车道的奖励 *** (原为 0.1)
    reward_staying_slow_penalty_scale = 0.1 # *** 新增: 未换到更快车道的惩罚 ***
    time_alive_reward = 0.01
    reward_comfort_penalty_scale = 0.05
    target_speed_factor = 0.95
    safe_distance_penalty_scale = 0.2
    min_buffer_dist_reward = 5.0
    time_gap_reward = 0.8
    # 奖励计算中考虑换道的最小安全距离
    min_safe_change_dist = 15.0

#####################
#   归一化          #
#####################
# RunningMeanStd 保持完全相同
class RunningMeanStd:
    """计算运行均值和标准差"""
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4, alpha: float = 0.001):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.alpha = alpha # 指数移动平均 alpha

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        # batch_count = x.shape[0] # Not strictly needed for EMA
        # EMA 更新
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1.0 - self.alpha) * self.var + self.alpha * batch_var
        # self.count += batch_count # Keep count for reference, but EMA drives updates

    @property
    def std(self) -> np.ndarray:
        # Add epsilon here too for safety, although var should be >= 0 with EMA
        return np.sqrt(np.maximum(self.var, 1e-8))

# 奖励缩放助手 - 现在归一化单步奖励
class RewardNormalizer:
    def __init__(self, gamma: float, epsilon: float = 1e-8, alpha: float = 0.001, clip: float = 10.0):
        self.ret_rms = RunningMeanStd(shape=(), alpha=alpha) # Use RunningMeanStd for reward std
        self.epsilon = epsilon
        self.clip = clip
        # Gamma not used directly here anymore, but kept for signature consistency

    def normalize(self, r: np.ndarray) -> np.ndarray:
        # Update running stats with current rewards
        # We expect r to be a single reward or a batch of rewards (1D array)
        if not isinstance(r, np.ndarray): # Ensure it's a numpy array
             r = np.array(r)
        if r.ndim == 0: # If single scalar reward, wrap in array
            r_update = np.array([r.item()]) # Use item() to get scalar if it's a 0-dim array
        else:
            r_update = r
        self.ret_rms.update(r_update)
        # Normalize: r / std
        norm_r = r / (self.ret_rms.std + self.epsilon) # Add epsilon here
        return np.clip(norm_r, -self.clip, self.clip)


#####################
#   工具函数         #
#####################
# get_available_port, kill_sumo_processes 保持完全相同
def get_available_port(start_port, end_port):
    """在指定范围内查找可用端口"""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError(f"在范围 [{start_port}, {end_port}] 内未找到可用端口。")

def kill_sumo_processes():
    """杀死任何残留的 SUMO 进程"""
    # print("尝试终止残留的 SUMO 进程...") # Reduce redundancy
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
        # if killed: print("已终止一个或多个 SUMO 进程。")
    except Exception as e: print(f"警告: 终止 SUMO 进程时出错: {e}")
    time.sleep(0.1) # Shorter sleep

# 线性衰减函数 (用于 PER beta)
def linear_decay(start_val, end_val, total_steps, current_step):
    """线性衰减计算"""
    if current_step >= total_steps:
        return end_val
    fraction = min(1.0, current_step / total_steps)
    return start_val + (end_val - start_val) * fraction

# --- Helper for Plotting Rolling Average ---
def calculate_rolling_average(data, window):
    if len(data) < window:
        return np.array([]) # Not enough data for a full window
    # Ensure data is numpy array of floats for convolve
    data_np = np.array(data, dtype=float)
    # Handle NaNs or Infs if they somehow appear in data (e.g., from rewards)
    data_np[~np.isfinite(data_np)] = 0 # Replace non-finite with 0 for averaging
    return np.convolve(data_np, np.ones(window) / window, mode='valid')

#####################
#   SUMO 环境封装    # (逻辑不变, 奖励计算不变)
#####################
class SumoEnv:
    def __init__(self, config: Config):
        self.config = config
        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_speed = 0.0 # 用于舒适度惩罚
        self.last_raw_state = np.zeros(config.state_dim) # 存储归一化前的原始状态
        self.last_norm_state = np.zeros(config.state_dim) # 存储归一化后的状态
        self.last_lane_idx = 0 # 存储上一个车道索引用于奖励计算
        self.num_lanes = config.num_train_lanes # Store number of lanes

        # Normalization (Applied inside environment)
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate, clip=config.reward_norm_clip) if config.normalize_rewards else None

        # Metrics
        self.reset_metrics()

    def reset_metrics(self):
        """重置回合指标"""
        self.change_lane_count = 0
        self.collision_occurred = False
        self.current_step = 0
        self.last_action = 0 # 上次执行的动作

    def _start_sumo(self):
        """启动 SUMO 实例并连接 TraCI"""
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.config.port_range[0], self.config.port_range[1])
        except IOError as e:
             print(f"错误: 无法找到可用端口: {e}")
             sys.exit(1)

        sumo_cmd = [
            self.config.sumo_binary, "-c", self.config.config_path,
            "--remote-port", str(self.traci_port),
            "--step-length", str(self.config.step_length),
            "--collision.check-junctions", "true",
            "--collision.action", "warn", # Use "warn" to detect collisions without stopping sim immediately
            "--time-to-teleport", "-1", # Disable teleporting
            "--no-warnings", "true",
            "--seed", str(np.random.randint(0, 10000))
        ]
        if self.config.sumo_binary == "sumo-gui":
             sumo_cmd.extend(["--start", "--quit-on-end"]) # Auto-start and close GUI

        try:
             stdout_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
        except FileNotFoundError:
             print(f"错误: SUMO 可执行文件 '{self.config.sumo_binary}' 未找到。")
             sys.exit(1)
        except Exception as e:
             print(f"错误: 无法启动 SUMO 进程: {e}")
             sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5)
                traci.init(self.traci_port)
                # print(f"✅ SUMO TraCI 已连接 (端口: {self.traci_port}).") # Reduce redundancy
                return
            except traci.exceptions.TraCIException:
                if attempt == connection_attempts - 1:
                    print("达到最大 TraCI 连接尝试次数。")
                    self._close()
                    raise ConnectionError(f"无法连接到 SUMO (端口: {self.traci_port})。")
            except Exception as e:
                print(f"连接 TraCI 时发生意外错误: {e}")
                self._close()
                raise ConnectionError(f"连接到 SUMO 时发生未知错误 (端口: {self.traci_port})。")

    def _add_ego_vehicle(self):
        """将 Ego 车辆添加到仿真中"""
        ego_route_id = "route_E0"
        if ego_route_id not in traci.route.getIDList():
             try: traci.route.add(ego_route_id, ["E0"])
             except traci.exceptions.TraCIException as e: raise RuntimeError(f"添加路径 '{ego_route_id}' 失败: {e}")

        if self.config.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.config.ego_type_id)
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0") # Red
                # Apply DQN-specific parameters if they differ, otherwise use defaults from 'car'
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcSpeedGain", "2.0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcCooperative", "0.5")
                traci.vehicletype.setParameter(self.config.ego_type_id, "jmIgnoreFoeProb", "0.1")
                traci.vehicletype.setParameter(self.config.ego_type_id, "carFollowModel", "IDM")
                traci.vehicletype.setParameter(self.config.ego_type_id, "minGap", "2.5")
            except traci.exceptions.TraCIException as e: print(f"警告: 设置 Ego 类型 '{self.config.ego_type_id}' 参数失败: {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            try: traci.vehicle.remove(self.config.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"警告: 移除残留 Ego 失败: {e}")

        try:
            start_lane = random.choice(range(self.num_lanes)) # Use internal num_lanes
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                              typeID=self.config.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max")

            wait_steps = int(2.0 / self.config.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                     ego_appeared = True; break
            if not ego_appeared:
                 print(f"警告: 在车道 {start_lane} 上添加 Ego 失败，尝试随机车道。")
                 # Fallback to random just in case specific lane fails
                 traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                                   typeID=self.config.ego_type_id, depart="now",
                                   departLane="random", departSpeed="max")
                 for _ in range(wait_steps):
                     traci.simulationStep()
                     if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                         ego_appeared = True; break
                 if not ego_appeared:
                    raise RuntimeError(f"Ego 车辆在 {wait_steps*2} 步内未出现。")

        except traci.exceptions.TraCIException as e:
            print(f"错误: 添加 Ego 车辆 '{self.config.ego_vehicle_id}' 失败: {e}")
            raise RuntimeError("添加 Ego 车辆失败。")
        except RuntimeError as e:
            print(f"运行时错误: {e}")
            self._close()
            raise


    def reset(self) -> np.ndarray:
        """为新回合重置环境，返回归一化状态"""
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
                 raw_state = self._get_raw_state()
                 self.last_raw_state = raw_state.copy()
                 self.last_lane_idx = np.clip(int(round(raw_state[1])), 0, self.num_lanes - 1)
                 norm_state = self._normalize_state(raw_state)
                 self.last_norm_state = norm_state.copy()
             except traci.exceptions.TraCIException as e:
                 print(f"警告: 在 reset 中的初始状态获取期间发生 TraCI 异常: {e}")
                 norm_state = np.zeros(self.config.state_dim) # Return default if error
             except IndexError:
                 print("警告: 访问 reset 中的初始状态时发生 IndexError。")
                 norm_state = np.zeros(self.config.state_dim)
        else:
             print("警告: 在 reset 中的 add/wait 后未立即找到 Ego 车辆。")
             # Attempt one more step to see if it appears
             try: traci.simulationStep()
             except traci.exceptions.TraCIException: pass
             if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                  return self.reset() # Try reset again if it just appeared
             else:
                  print("错误: Ego 车辆在 reset 后仍然不存在。")
                  # Return a default state, but this indicates a problem
                  norm_state = np.zeros(self.config.state_dim)


        return norm_state # Return normalized state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """使用运行均值/标准差归一化状态。"""
        if self.obs_normalizer:
            self.obs_normalizer.update(state.reshape(1, -1)) # Update with single observation
            norm_state = (state - self.obs_normalizer.mean) / self.obs_normalizer.std
            norm_state = np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip)
            # Check for NaN/Inf *after* normalization
            if np.any(np.isnan(norm_state)) or np.any(np.isinf(norm_state)):
                # print(f"警告: 归一化状态中检测到 NaN/Inf。返回上一个有效的归一化状态。") # Reduce spam
                return self.last_norm_state.copy() # Use last known good normalized state
            return norm_state.astype(np.float32)
        else:
            return state # Return raw state if normalization off


    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """获取最近车辆的距离和相对速度 (未更改)"""
        max_dist = self.config.max_distance
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        try:
            veh_ids = traci.vehicle.getIDList()
            if ego_id not in veh_ids: return infos
        except traci.exceptions.TraCIException:
            return infos # Error getting vehicle list

        try:
            ego_road = traci.vehicle.getRoadID(ego_id)
            if not ego_road or not ego_road.startswith("E"): return infos # Check on expected edge
        except traci.exceptions.TraCIException:
             return infos # Error getting ego road ID

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                if traci.vehicle.getRoadID(veh_id) != ego_road: continue

                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)

                # Ensure valid lane index before comparing
                if not (0 <= veh_lane < self.num_lanes): continue

                dx = veh_pos[0] - ego_pos[0] # Assume X-axis alignment primarily
                distance = abs(dx) # Using longitudinal distance for simplicity

                if distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed # Positive if ego is faster

                # Determine relative position based on lane and dx
                if veh_lane == ego_lane: # Same lane
                    if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                elif veh_lane == ego_lane - 1: # Left lane
                    if dx > 0 and distance < infos['left_front'][0]: infos['left_front'] = (distance, rel_speed)
                    elif dx <= 0 and distance < infos['left_back'][0]: infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1: # Right lane
                     if dx > 0 and distance < infos['right_front'][0]: infos['right_front'] = (distance, rel_speed)
                     elif dx <= 0 and distance < infos['right_back'][0]: infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue # Skip vehicle if TraCI error occurs
        return infos

    def _get_raw_state(self) -> np.ndarray:
        """获取当前环境状态 (归一化之前的原始值)"""
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id

        try:
            if ego_id not in traci.vehicle.getIDList():
                # print(f"警告: _get_raw_state 无法找到 ego {ego_id}。返回最后已知的原始状态。") # Reduce spam
                return self.last_raw_state # Return last known raw state if ego disappeared
        except traci.exceptions.TraCIException:
            # print(f"警告: 检查 ego 存在性时 TraCI 错误。返回最后已知的原始状态。") # Reduce spam
            return self.last_raw_state

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road or not current_road.startswith("E"):
                # print(f"警告: Ego 不在预期道路上 ({current_road})。返回最后已知的原始状态。") # Reduce spam
                return self.last_raw_state # Not on expected edge

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)

            # Validate lane index
            if not (0 <= ego_lane < self.num_lanes):
                 # print(f"警告: _get_raw_state 中无效的车道索引 {ego_lane}。尝试重新获取。") # Reduce spam
                 time.sleep(0.05)
                 if ego_id in traci.vehicle.getIDList():
                      try: ego_lane = traci.vehicle.getLaneIndex(ego_id)
                      except traci.exceptions.TraCIException: ego_lane = self.last_lane_idx # Fallback
                 else: ego_lane = self.last_lane_idx # Fallback if gone
                 if not (0 <= ego_lane < self.num_lanes):
                      # print(f"警告: _get_raw_state 中重新获取后车道索引仍然无效 ({ego_lane})。使用最后有效的索引 {self.last_lane_idx}。") # Reduce spam
                      ego_lane = self.last_lane_idx
                 ego_lane = np.clip(ego_lane, 0, self.num_lanes - 1)


            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (self.num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # --- Raw State Features ---
            state[0] = ego_speed
            state[1] = float(ego_lane) # Keep lane index as float for consistency/normalization
            state[2] = min(surround_info['front'][0], self.config.max_distance)
            state[3] = surround_info['front'][1] # Relative speed (don't clip here)
            state[4] = min(surround_info['left_front'][0], self.config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.config.max_distance)
            state[7] = min(surround_info['right_front'][0], self.config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.config.max_distance)
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"警告: 在原始状态计算中检测到 NaN 或 Inf。使用最后有效的原始状态。")
                return self.last_raw_state.copy()

            self.last_raw_state = state.copy() # Store latest valid raw state

        except traci.exceptions.TraCIException as e:
            is_not_known = "Vehicle '" + ego_id + "' is not known" in str(e)
            is_teleporting = "teleporting" in str(e).lower()
            if not is_not_known and not is_teleporting:
                print(f"警告: 获取 {ego_id} 的原始状态时发生 TraCI 错误: {e}。返回最后已知的原始状态。")
            return self.last_raw_state.copy()
        except Exception as e:
            print(f"警告: 获取 {ego_id} 的原始状态时发生未知错误: {e}。返回最后已知的原始状态。"); traceback.print_exc()
            return self.last_raw_state.copy()

        return state # Return raw state


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """执行一个动作，返回 (next_normalized_state, normalized_reward, done, next_lane_index)"""
        done = False
        raw_reward = 0.0 # Reward before normalization
        next_lane_index = self.last_lane_idx # Initialize with current lane
        ego_id = self.config.ego_vehicle_id
        self.last_action = action # Store action taken

        try:
            if ego_id not in traci.vehicle.getIDList():
                 # print(f"警告: 步骤开始时未找到 Ego {ego_id}。假设碰撞。") # Reduce spam
                 self.collision_occurred = True; done = True
                 coll_reward_raw = self.config.reward_collision
                 norm_coll_reward = self.reward_normalizer.normalize(np.array([coll_reward_raw]))[0] if self.reward_normalizer else coll_reward_raw
                 return self.last_norm_state.copy(), norm_coll_reward, True, self.last_lane_idx
        except traci.exceptions.TraCIException:
            # print(f"警告: 检查步骤开始时 Ego 存在性时 TraCI 错误。假设碰撞。") # Reduce spam
            self.collision_occurred = True; done = True
            coll_reward_raw = self.config.reward_collision
            norm_coll_reward = self.reward_normalizer.normalize(np.array([coll_reward_raw]))[0] if self.reward_normalizer else coll_reward_raw
            return self.last_norm_state.copy(), norm_coll_reward, True, self.last_lane_idx


        try:
            previous_lane_idx = self.last_lane_idx
            current_lane = previous_lane_idx # Start with the known lane index

            # 1. Execute Action if valid
            lane_change_requested = False
            if action == 1: # Try Left
                if current_lane > 0 and traci.vehicle.couldChangeLane(ego_id, -1):
                     traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0) # Request change
                     lane_change_requested = True
            elif action == 2: # Try Right
                if current_lane < (self.num_lanes - 1) and traci.vehicle.couldChangeLane(ego_id, 1):
                     traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0) # Request change
                     lane_change_requested = True
            # If action is 0 (Keep Lane), do nothing explicitly

            # 2. Simulation Step
            traci.simulationStep()
            self.current_step += 1

            # 3. Check State and Calculate Reward *after* step
            if ego_id not in traci.vehicle.getIDList():
                # print(f"警告: 步骤后未找到 Ego {ego_id}。假设碰撞。") # Reduce spam
                self.collision_occurred = True; done = True
                raw_reward = self.config.reward_collision
                next_norm_state = self.last_norm_state.copy() # Return last known normalized state
                next_lane_index = self.last_lane_idx # Use last known lane
                normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward
                return next_norm_state, normalized_reward, done, next_lane_index

            # Explicit collision check
            collisions = traci.simulation.getCollisions()
            ego_collided = False
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    # print(f"碰撞检测: {col.collider} vs {col.victim}") # Debug collision
                    self.collision_occurred = True; ego_collided = True; done = True
                    raw_reward = self.config.reward_collision
                    break

            # Get next state (raw first, then normalized)
            next_raw_state = self._get_raw_state()
            next_norm_state = self._normalize_state(next_raw_state)
            # Get lane index *after* the step from the new raw state
            next_lane_index = np.clip(int(round(next_raw_state[1])), 0, self.num_lanes - 1)

            # Calculate reward based on state *after* step, unless collision happened
            if not self.collision_occurred:
                 # Extract necessary info from *raw* state for reward calc
                 current_speed_after_step = next_raw_state[0]
                 front_dist_after_step = next_raw_state[2]
                 left_front_dist = next_raw_state[4]; left_back_dist = next_raw_state[6]
                 right_front_dist = next_raw_state[7]; right_back_dist = next_raw_state[9]
                 can_change_left_after_step = next_raw_state[10] > 0.5
                 can_change_right_after_step = next_raw_state[11] > 0.5

                 actual_lane_change = (previous_lane_idx != next_lane_index)
                 if actual_lane_change: self.change_lane_count += 1

                 # Pass necessary state components to reward function
                 raw_reward = self._calculate_reward(action_taken=action, # Pass the action we attempted
                                                 current_speed=current_speed_after_step,
                                                 current_lane=next_lane_index, # Use lane *after* step
                                                 front_dist=front_dist_after_step,
                                                 previous_lane=previous_lane_idx, # Pass previous lane
                                                 can_change_left=can_change_left_after_step,
                                                 can_change_right=can_change_right_after_step,
                                                 left_front_dist=left_front_dist, left_back_dist=left_back_dist,
                                                 right_front_dist=right_front_dist, right_back_dist=right_back_dist,
                                                 lane_change_requested=lane_change_requested) # Pass if we requested change


            # 4. Check other termination conditions
            if traci.simulation.getTime() >= 3600: done = True # Simulation time limit
            if self.current_step >= self.config.max_steps: done = True # Step limit per episode

        except traci.exceptions.TraCIException as e:
            is_known_error = "Vehicle '" + ego_id + "' is not known" in str(e) or "teleporting" in str(e).lower()
            if not is_known_error: print(f"错误: 在步骤 {self.current_step} 期间发生 TraCI 异常: {e}")
            if is_known_error or (ego_id not in traci.vehicle.getIDList() if not is_known_error else False):
                self.collision_occurred = True
            raw_reward = self.config.reward_collision; done = True
            next_norm_state = self.last_norm_state.copy(); next_lane_index = self.last_lane_idx
        except Exception as e:
            print(f"错误: 在步骤 {self.current_step} 期间发生未知异常: {e}"); traceback.print_exc()
            done = True; raw_reward = self.config.reward_collision
            self.collision_occurred = True; next_norm_state = self.last_norm_state.copy(); next_lane_index = self.last_lane_idx

        # 5. Update last state info for next step
        # Check if next_raw_state is valid before updating last_speed
        if not (np.any(np.isnan(next_raw_state)) or np.any(np.isinf(next_raw_state))):
            self.last_speed = next_raw_state[0]
        # Use the potentially corrected next_lane_index
        self.last_lane_idx = next_lane_index
        # Ensure we store a valid normalized state
        if not (np.any(np.isnan(next_norm_state)) or np.any(np.isinf(next_norm_state))):
            self.last_norm_state = next_norm_state.copy() # Store normalized state

        # *** Normalize Reward before returning ***
        normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward
        # Ensure reward is finite
        if not np.isfinite(normalized_reward):
            # print(f"警告: 归一化奖励非有限 ({normalized_reward})。在回合 {self.current_step} 设置为 0。") # Reduce spam
            normalized_reward = 0.0

        # Return normalized state, normalized reward, done flag, and next lane index
        return self.last_norm_state.copy(), normalized_reward, done, next_lane_index # Return the stored last_norm_state


    def _calculate_reward(self, action_taken: int, current_speed: float, current_lane: int,
                           front_dist: float, previous_lane: int,
                           can_change_left: bool, can_change_right: bool,
                           left_front_dist: float, left_back_dist: float, # Added back distances
                           right_front_dist: float, right_back_dist: float, # Added back distances
                           lane_change_requested: bool) -> float: # Added flag
        """Calculate reward based on state *after* the step, including new penalties/bonuses."""
        # Collision reward is handled in step function before calling this
        if self.collision_occurred: return 0.0

        try:
            current_lane = np.clip(current_lane, 0, self.num_lanes - 1)
            previous_lane = np.clip(previous_lane, 0, self.num_lanes - 1)

            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            # --- Reward Components ---
            speed_diff = abs(current_speed - target_speed)
            # Use max speed for normalization to prevent division by zero or tiny target speeds if stopped
            norm_target_speed = max(lane_max_speed * 0.5, 1.0) # Avoid very small denominators
            speed_reward = np.exp(- (speed_diff / norm_target_speed)**2 ) * self.config.reward_high_speed_scale

            low_speed_penalty = 0.0
            low_speed_threshold = target_speed * 0.6
            if current_speed < low_speed_threshold and target_speed > 1.0: # Avoid penalty when target is near zero
                low_speed_penalty = (current_speed / max(low_speed_threshold, 1.0) - 1.0) * self.config.reward_low_speed_penalty_scale

            lane_change_actual = (current_lane != previous_lane)
            lane_change_reward_penalty = 0.0 # Combined term
            if lane_change_actual:
                lane_change_reward_penalty += self.config.reward_lane_change_penalty
                # Bonus only if moving to a strictly faster potential lane
                if self.config.lane_max_speeds[current_lane] > self.config.lane_max_speeds[previous_lane]:
                     lane_change_reward_penalty += self.config.reward_faster_lane_bonus

            staying_slow_penalty = 0.0
            if not lane_change_actual and action_taken == 0: # Penalize only if agent chose 'Keep Lane'
                left_lane_idx = current_lane - 1
                # Check if left lane exists and is potentially faster
                if left_lane_idx >= 0 and self.config.lane_max_speeds[left_lane_idx] > lane_max_speed:
                    # Check if a safe change was possible
                    if can_change_left and left_front_dist > self.config.min_safe_change_dist and left_back_dist > self.config.min_safe_change_dist * 0.6:
                        staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale
                # Check right lane only if left wasn't a viable faster option
                elif staying_slow_penalty == 0.0:
                    right_lane_idx = current_lane + 1
                    # Check if right lane exists and is potentially faster
                    if right_lane_idx < self.num_lanes and self.config.lane_max_speeds[right_lane_idx] > lane_max_speed:
                         # Check if a safe change was possible
                         if can_change_right and right_front_dist > self.config.min_safe_change_dist and right_back_dist > self.config.min_safe_change_dist * 0.6:
                             staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale

            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < self.config.max_distance and front_dist < min_safe_dist:
                 # Linear penalty: 0 at min_safe_dist, scales down to -safe_distance_penalty_scale at 0 distance
                 safety_dist_penalty = (front_dist / max(min_safe_dist, 1e-6) - 1.0) * self.config.safe_distance_penalty_scale

            comfort_penalty = 0.0
            acceleration = (current_speed - self.last_speed) / self.config.step_length
            harsh_braking_threshold = -3.0 # m/s^2
            if acceleration < harsh_braking_threshold:
                 # Linear penalty based on how much braking exceeds threshold
                 comfort_penalty = (acceleration - harsh_braking_threshold) * self.config.reward_comfort_penalty_scale # Negative

            time_alive = self.config.time_alive_reward

            total_reward = (speed_reward +
                            low_speed_penalty +
                            lane_change_reward_penalty +
                            staying_slow_penalty +
                            safety_dist_penalty +
                            comfort_penalty +
                            time_alive)

            # Ensure reward is not NaN/Inf before returning
            if not np.isfinite(total_reward):
                # print(f"警告: 计算的原始奖励非有限。Speed={current_speed}, Lane={current_lane}, FrontD={front_dist}, PrevLane={previous_lane}") # Reduce spam
                total_reward = 0.0 # Return 0 if calculation resulted in issue

            return total_reward

        except IndexError as e_idx:
            print(f"警告: 计算奖励时发生 IndexError (可能是无效的车道索引 {current_lane} 或 {previous_lane}): {e_idx}。返回 0。")
            traceback.print_exc()
            return 0.0
        except Exception as e:
            print(f"警告: 计算奖励时出错: {e}。返回 0。"); traceback.print_exc()
            return 0.0

    def _close(self):
        """关闭 SUMO 实例和 TraCI 连接 (与 ppo.py 完全相同)"""
        if self.sumo_process:
            try: traci.close()
            except traci.exceptions.FatalTraCIError: pass # Ignore if connection already closed
            except Exception: pass # Ignore other errors on close
            finally:
                try:
                    if self.sumo_process.poll() is None: self.sumo_process.terminate(); self.sumo_process.wait(timeout=2)
                except subprocess.TimeoutExpired: self.sumo_process.kill(); self.sumo_process.wait(timeout=1)
                except Exception as e: print(f"警告: SUMO 终止期间出错: {e}")
                self.sumo_process = None; self.traci_port = None; time.sleep(0.1)
        else: self.traci_port = None


#####################
#   DQN 组件         # (逻辑不变)
#####################

# --- N-Step Transition (添加了 next_lane_index) ---
NStepTransition = namedtuple('NStepTransition', ('state', 'action', 'reward', 'next_state', 'done', 'next_lane_index'))

# --- Experience for Replay Buffer (PER + N-Step) (添加了 next_lane_index) ---
Experience = namedtuple('Experience', ('state', 'action', 'n_step_reward', 'next_state', 'done', 'n', 'next_lane_index'))


# --- SumTree for Prioritized Replay --- (未更改)
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
        # Robustness check for data index
        if dataIdx < 0 or dataIdx >= self.capacity:
            print(f"警告: 从 SumTree 检索到无效的 dataIdx {dataIdx} (tree idx {idx}, s={s}, total={self.total()})。返回虚拟数据。")
            dummy_state = np.zeros(Config.state_dim)
            return (idx, 0.0, Experience(dummy_state, 0, 0, dummy_state, True, 1, 0))

        data = self.data[dataIdx]
        if not isinstance(data, Experience): # Robustness check
            # print(f"警告: 在 SumTree 的 dataIdx {dataIdx} (tree idx {idx}) 处发现损坏的数据。 data: {data}。返回虚拟数据。") # Reduce spam
            dummy_state = np.zeros(Config.state_dim) # Use state_dim from Config
            # Return the original index but with dummy data and low priority (tree[idx])
            return (idx, self.tree[idx], Experience(dummy_state, 0, 0, dummy_state, True, 1, 0))
        return (idx, self.tree[idx], data)
    def __len__(self): return self.n_entries

# --- Prioritized Replay Buffer (PER + N-Step) --- (逻辑未更改，依赖于 Experience 元组)
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
        if self.tree.total() <= 0 or batch_size <= 0 or segment <= 0: # Avoid division by zero or invalid segment
             # print("警告：SumTree 总和为零或 batch_size 为零，无法采样。返回空列表。") # Reduce spam
             return [], np.array([]), np.array([])

        for i in range(batch_size):
            attempt = 0
            successful_sample = False
            while attempt < 10: # Retry sampling a few times if invalid data encountered
                a = segment * i; b = segment * (i + 1);
                # Ensure s is strictly less than total() if total() > 0
                s = random.uniform(a, min(b, self.tree.total()))
                # Clamp s to be within valid range [0, total)
                s = max(0.0, min(s, self.tree.total() - 1e-9)) if self.tree.total() > 0 else 0.0

                try:
                    (idx, p, data) = self.tree.get(s)

                    # Check if data is valid *after* getting it
                    if not isinstance(data, Experience) or not isinstance(data.state, np.ndarray) or data.state.shape != (self.config.state_dim,):
                        # print(f"警告: 在索引 {idx} 处采样到无效数据 (类型: {type(data)}), tree 总和 {self.tree.total()}, segment [{a:.4f},{b:.4f}], s {s:.4f}。重试采样 ({attempt+1}/10)...") # Reduce spam
                        attempt += 1; time.sleep(0.001)
                        # Recalculate segment in case total changed (unlikely but possible)
                        segment = self.tree.total() / batch_size if batch_size > 0 and self.tree.total() > 0 else 0
                        if segment <= 0: break # Stop retrying if segment becomes invalid
                        continue # Retry loop

                    experiences.append(data); indices[i] = idx;
                    prob = p / (self.tree.total() + 1e-9) # Add epsilon for stability
                    # Ensure probability is positive
                    prob = max(prob, 1e-9)

                    is_weights[i] = np.power(self.tree.n_entries * prob, -beta)
                    successful_sample = True
                    break # Sample successful, exit retry loop
                except Exception as e:
                    print(f"为 s={s} (range [{a},{b}]), idx={idx}, tree total={self.tree.total()} 执行 SumTree.get 或处理时出错: {e}")
                    traceback.print_exc()
                    attempt += 1; time.sleep(0.001)
                    segment = self.tree.total() / batch_size if batch_size > 0 and self.tree.total() > 0 else 0
                    if segment <= 0: break # Stop retrying if segment becomes invalid

            if not successful_sample:
                 # print(f"尝试 {attempt} 次后未能从 SumTree 采样到有效的 Experience 数据。 Tree total: {self.tree.total()}, n_entries: {self.tree.n_entries}") # Reduce spam
                 # If even one sample fails after retries, return empty to avoid partial batch issues
                 return [], np.array([]), np.array([])


        # Normalize IS weights, add epsilon for stability before division
        max_weight = is_weights.max()
        is_weights /= (max_weight + 1e-8)

        # Sanity check weights
        if np.any(~np.isfinite(is_weights)):
            print(f"警告: IS 权重包含 NaN 或 Inf。 Weights: {is_weights}")
            # Handle this case, e.g., replace NaNs/Infs or return empty
            is_weights[~np.isfinite(is_weights)] = 1.0 # Replace with neutral weight

        return experiences, is_weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + self.config.per_epsilon # Use absolute TD error, add epsilon
        for idx, priority in zip(indices, priorities):
            # Check index validity relative to the tree structure
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                # print(f"警告: 提供给 update_priorities 的索引 {idx} 无效 (capacity={self.tree.capacity})。跳过。") # Reduce spam
                continue
            if not np.isfinite(priority):
                 # print(f"警告: 索引 {idx} 的优先级 {priority} 非有限。进行裁剪。") # Reduce log spam
                 priority = self.max_priority # Clip to known max priority
            elif priority <= 0:
                 # print(f"警告: 索引 {idx} 的优先级 {priority} <= 0。使用 epsilon。") # Reduce spam
                 priority = self.config.per_epsilon

            self.max_priority = max(self.max_priority, priority)
            try:
                self.tree.update(idx, priority ** self.alpha) # Update using the tree index
            except Exception as e:
                print(f"更新 SumTree 索引 {idx} 优先级 {priority} 时出错: {e}")
                # Optionally try to recover or just skip the update for this index
                continue
    def __len__(self) -> int: return self.tree.n_entries

# --- Noisy Linear Layer --- (未更改)
class NoisyLinear(nn.Module):
    """Noisy Linear layer with Factorized Gaussian Noise"""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Ensure noise tensors are on the same device as parameters
            self.reset_noise() # Reset noise at each forward pass during training
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon.to(self.weight_sigma.device))
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon.to(self.bias_sigma.device))
        else: # Use mean weights/biases during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# --- Q-Network (Modified for C51 and Noisy Nets) --- (网络结构未更改)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, config: Config):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = config.num_atoms
        self.use_noisy = config.use_noisy_nets
        self.sigma_init = config.noisy_sigma_init

        linear = lambda in_f, out_f: NoisyLinear(in_f, out_f, self.sigma_init) if self.use_noisy else nn.Linear(in_f, out_f)

        self.feature_layer = nn.Sequential(
            linear(state_dim, hidden_size), nn.ReLU(),
        )
        # Dueling Architecture streams
        self.value_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.num_atoms) # Outputs V(s) distribution (num_atoms logits)
        )
        self.advantage_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.action_dim * self.num_atoms) # Outputs A(s,a) distributions (action_dim * num_atoms logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.feature_layer(x)

        values_logits = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantages_logits = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)

        # Combine V and A streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Apply this combination to the logits *before* the softmax
        q_logits = values_logits + advantages_logits - advantages_logits.mean(dim=1, keepdim=True)

        # Apply softmax to get probability distributions for each action
        q_probs = F.softmax(q_logits, dim=2) # Softmax over atoms dimension

        # Clamp probabilities for numerical stability (optional but good practice)
        q_probs = q_probs.clamp(min=1e-8)
        # Re-normalize after clamping (optional, softmax should already sum to 1)
        # q_probs /= q_probs.sum(dim=2, keepdim=True)

        return q_probs # Return distributions [batch, action, atoms]

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


#####################
#    DQN Agent      # (逻辑不变)
#####################
class DQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms) # Distribution supports (atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1) if self.num_atoms > 1 else 0 # Distance between atoms
        self.num_lanes = config.num_train_lanes # Store num lanes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"使用设备: {self.device}") # Already printed in main
        self.support = self.support.to(self.device) # Move supports to device

        self.policy_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.initial_learning_rate, eps=1e-5) # Adam epsilon

        # Initialize Replay Buffer
        if config.use_per:
             self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size, config.per_alpha, config)
        else:
             print("警告: PER 已禁用，使用标准回放缓冲区。")
             # Standard Experience Replay Buffer (if PER is off)
             self.replay_buffer = deque([], maxlen=config.replay_buffer_size)
             # Need to adapt push/sample logic if not using PER
             raise NotImplementedError("Standard Replay Buffer push/sample logic not fully implemented here, designed for PER.")


        self.train_step_count = 0
        self.loss_history = [] # Track loss for logging

    def get_action(self, normalized_state: np.ndarray, current_lane_idx: int) -> int:
        """Choose action based on expected Q-values from (noisy) policy net"""
        self.policy_net.eval() # Set to eval mode for action selection
        # No need to reset noise here if NoisyLinear resets noise only during training forward pass
        # if self.config.use_noisy_nets: self.policy_net.reset_noise() # Keep this if NoisyLinear doesn't reset automatically

        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(state_tensor) # [1, action_dim, num_atoms]
            # Calculate expected Q-values by weighting support by probabilities
            expected_q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2) # [1, action_dim]

            q_values_masked = expected_q_values.clone()
            # Ensure current_lane_idx is valid
            current_lane_idx = np.clip(current_lane_idx, 0, self.num_lanes - 1)
            # Mask invalid actions: Cannot change left from lane 0
            if current_lane_idx == 0: q_values_masked[0, 1] = -float('inf')
            # Cannot change right from the rightmost lane
            if current_lane_idx >= self.num_lanes - 1: q_values_masked[0, 2] = -float('inf')

            action = q_values_masked.argmax().item()

        self.policy_net.train() # Set back to train mode
        return action

    def update(self, global_step: int) -> Optional[float]:
        """Update Q-network using a batch from replay buffer"""
        # Only update if buffer has enough samples and learning has started
        if len(self.replay_buffer) < max(self.config.learning_starts, self.config.batch_size):
            return None

        current_beta = linear_decay(self.config.per_beta_start, self.config.per_beta_end,
                                    self.config.per_beta_annealing_steps, global_step)

        try: # Sample from buffer
            if self.config.use_per:
                experiences, is_weights, indices = self.replay_buffer.sample(self.config.batch_size, current_beta)
                if not experiences:
                    # print("警告: 从 PER 采样失败。跳过更新。") # Reduce spam
                    return None # Handle empty sample case from PER
            else:
                # Standard replay buffer sampling (if implemented)
                # experiences = random.sample(self.replay_buffer, self.config.batch_size)
                # is_weights = np.ones(self.config.batch_size); indices = None
                raise NotImplementedError("Standard Replay Buffer sampling logic needed if PER is off.")
        except RuntimeError as e:
            print(f"从回放缓冲区采样时出错: {e}。跳过更新。"); return None
        except Exception as e_sample:
             print(f"从回放缓冲区采样时发生意外错误: {e_sample}。跳过更新。"); traceback.print_exc(); return None


        # Unpack batch and convert to tensors
        try:
            batch = Experience(*zip(*experiences))

            # Check for None or invalid states/rewards before converting
            valid_idx = [i for i, s in enumerate(batch.state) if s is not None and isinstance(s, np.ndarray) and s.shape==(self.state_dim,)]
            if len(valid_idx) != self.config.batch_size:
                 # print(f"警告: 采样批次中包含无效状态 ({self.config.batch_size - len(valid_idx)} / {self.config.batch_size})。跳过更新。") # Reduce spam
                 # Optionally update priorities of invalid ones to min?
                 return None

            states_np = np.array([batch.state[i] for i in valid_idx], dtype=np.float32)
            actions_np = np.array([batch.action[i] for i in valid_idx], dtype=np.int64) # Use int64 for LongTensor
            n_step_rewards_np = np.array([batch.n_step_reward[i] for i in valid_idx], dtype=np.float32)
            next_states_np = np.array([batch.next_state[i] for i in valid_idx], dtype=np.float32)
            dones_np = np.array([batch.done[i] for i in valid_idx], dtype=bool)
            n_np = np.array([batch.n[i] for i in valid_idx], dtype=np.float32) # Keep n as float for gamma calculation
            next_lane_indices_np = np.array([batch.next_lane_index[i] for i in valid_idx], dtype=np.int64) # Use int64

            # Ensure rewards are finite
            if not np.all(np.isfinite(n_step_rewards_np)):
                # print(f"警告: 批次奖励包含 NaN/Inf。在更新前将其裁剪/替换。") # Reduce spam
                n_step_rewards_np = np.nan_to_num(n_step_rewards_np, nan=0.0, posinf=self.config.reward_norm_clip, neginf=-self.config.reward_norm_clip)


            states = torch.from_numpy(states_np).to(self.device)
            actions = torch.from_numpy(actions_np).to(self.device)
            rewards = torch.from_numpy(n_step_rewards_np).to(self.device)
            next_states = torch.from_numpy(next_states_np).to(self.device)
            dones = torch.from_numpy(dones_np).to(self.device)
            is_weights_tensor = torch.from_numpy(np.array(is_weights, dtype=np.float32)).to(self.device) # Weights correspond to original sample
            gammas = torch.pow(self.config.gamma, torch.from_numpy(n_np).to(self.device)) # Calculate gamma^n
            next_lane_indices = torch.from_numpy(next_lane_indices_np).to(self.device)

        except Exception as e_tensor:
             print(f"将批次转换为张量时出错: {e_tensor}"); traceback.print_exc(); return None


        # --- Compute Target Distribution (C51 logic) ---
        with torch.no_grad():
            # if self.config.use_noisy_nets: self.target_net.reset_noise() # NoisyLinear handles eval mode
            next_dist_target = self.target_net(next_states) # [batch, action, atoms]

            # --- Double DQN Action Selection ---
            if self.config.use_double_dqn:
                # if self.config.use_noisy_nets: self.policy_net.reset_noise() # NoisyLinear handles eval mode
                # Use policy_net (in eval mode temporarily) to select best actions
                self.policy_net.eval()
                next_dist_policy = self.policy_net(next_states)
                self.policy_net.train() # Switch back to train mode

                next_expected_q_policy = (next_dist_policy * self.support.view(1, 1, -1)).sum(dim=2) # [batch, action]
                # Mask actions based on next_lane_indices before argmax
                q_values_masked = next_expected_q_policy.clone()
                q_values_masked[next_lane_indices == 0, 1] = -float('inf') # Mask left change if in lane 0
                q_values_masked[next_lane_indices >= self.num_lanes - 1, 2] = -float('inf') # Mask right change if in last lane
                best_next_actions = q_values_masked.argmax(dim=1) # [batch]
            else:
                 # Standard DQN: Use target_net to select best actions
                 next_expected_q_target = (next_dist_target * self.support.view(1, 1, -1)).sum(dim=2) # [batch, action]
                 # Mask actions based on next_lane_indices before argmax
                 q_values_masked = next_expected_q_target.clone()
                 q_values_masked[next_lane_indices == 0, 1] = -float('inf') # Mask left change if in lane 0
                 q_values_masked[next_lane_indices >= self.num_lanes - 1, 2] = -float('inf') # Mask right change if in last lane
                 best_next_actions = q_values_masked.argmax(dim=1) # [batch]

            # Get the distribution corresponding to the chosen best actions from the target network
            best_next_dist = next_dist_target[torch.arange(states.size(0)), best_next_actions, :] # [batch, atoms]

            # --- Project Target Distribution (C51 Distributional Bellman Update) ---
            # Compute the projected support Tz = R + gamma^n * Z_next (element-wise)
            # Unsqueeze rewards and gammas for broadcasting with support
            Tz = rewards.unsqueeze(1) + gammas.unsqueeze(1) * self.support.unsqueeze(0) * (~dones).unsqueeze(1).float() # [batch, atoms]
            # Clamp the projected support into the predefined range [V_min, V_max]
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            # Calculate the indices and weights for projecting onto the original support
            if self.delta_z > 0:
                b = (Tz - self.v_min) / self.delta_z # Normalized position on the support grid [batch, atoms]
                lower_idx = b.floor().long()
                upper_idx = b.ceil().long()
                # Handle cases where Tz is exactly on an atom (b is integer)
                # In this case, lower_idx == upper_idx, lower_weight=1, upper_weight=0 (or vice versa)
                lower_idx.clamp_(min=0, max=self.num_atoms - 1)
                upper_idx.clamp_(min=0, max=self.num_atoms - 1)

                # Calculate projection weights (linear interpolation)
                lower_weight = (upper_idx.float() - b).clamp(0, 1) # Weight for the lower atom index
                upper_weight = (b - lower_idx.float()).clamp(0, 1) # Weight for the upper atom index
            else: # Handle single atom case (num_atoms = 1)
                lower_idx = torch.zeros_like(Tz).long()
                upper_idx = torch.zeros_like(Tz).long()
                lower_weight = torch.ones_like(Tz)
                upper_weight = torch.zeros_like(Tz)


            # Initialize the projected target distribution
            target_dist_projected = torch.zeros_like(best_next_dist) # [batch, atoms]

            # Distribute the probability mass p(s', a*) from best_next_dist to the target atoms
            # Use index_add_ for efficient distribution (or index_put_ with accumulation)
            batch_indices = torch.arange(states.size(0), device=self.device)

            # Flatten indices and weights for index_add_
            flat_lower_idx = lower_idx.flatten() # [batch * atoms]
            flat_upper_idx = upper_idx.flatten() # [batch * atoms]
            flat_best_next_dist = best_next_dist.flatten() # [batch * atoms]
            flat_lower_weight = lower_weight.flatten() # [batch * atoms]
            flat_upper_weight = upper_weight.flatten() # [batch * atoms]
            flat_batch_indices = batch_indices.unsqueeze(1).expand(-1, self.num_atoms).flatten() # [batch * atoms]

            # Use index_put_ which is often safer with autograd if accumulation is intended
            target_dist_projected.index_put_((flat_batch_indices, flat_lower_idx), flat_best_next_dist * flat_lower_weight, accumulate=True)
            target_dist_projected.index_put_((flat_batch_indices, flat_upper_idx), flat_best_next_dist * flat_upper_weight, accumulate=True)

            # Ensure target distribution sums to 1 (optional, should be close)
            # target_dist_projected /= target_dist_projected.sum(dim=1, keepdim=True).clamp(min=1e-8)


        # --- Compute Current Distribution for chosen actions ---
        # if self.config.use_noisy_nets: self.policy_net.reset_noise() # NoisyLinear handles train mode
        current_dist_all_actions = self.policy_net(states) # [batch, action, atoms]
        # Select the distribution corresponding to the action actually taken in the batch
        current_dist = current_dist_all_actions[torch.arange(states.size(0)), actions, :] # [batch, atoms]

        # --- Compute Loss (Cross-Entropy between Target and Current Distributions) ---
        # Ensure current_dist has small epsilon for log stability
        elementwise_loss = -(target_dist_projected.detach() * (current_dist + 1e-8).log()).sum(dim=1) # [batch]
        # Detach target_dist_projected as it's the target, we don't backprop through it

        # Calculate TD errors for PER (using the elementwise loss before weighting/mean)
        td_errors = elementwise_loss.detach() # [batch]

        # Update priorities in PER buffer if enabled
        if self.config.use_per and indices is not None:
             # Ensure td_errors are finite before updating
             td_errors_np = td_errors.cpu().numpy()
             if np.all(np.isfinite(td_errors_np)):
                 self.replay_buffer.update_priorities(indices, td_errors_np)
             else:
                 print(f"警告: TD 错误包含 NaN/Inf。跳过优先级更新。Errors: {td_errors_np}")


        # Apply Importance Sampling (IS) weights and compute the mean loss
        loss = (elementwise_loss * is_weights_tensor).mean()

        # --- Optimization Step ---
        self.optimizer.zero_grad(); loss.backward()
        # Clip gradients to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.loss_history.append(loss.item()) # Append loss per step

        # --- Update Target Network ---
        if self.train_step_count % self.config.target_update_freq == 0:
            # print(f"--- 更新目标网络 @ 步骤 {self.train_step_count} ---") # Optional log
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

#####################
#   主训练循环       #
#####################
def main():
    config = Config()

    # --- Basic Checks & Setup ---
    required_files = [config.config_path, "a.net.xml", "a.rou.xml"]
    abort_training = False
    for f in required_files:
        if not os.path.exists(f): print(f"错误: 未找到所需的 SUMO 文件: {f}"); abort_training = True
    if abort_training: sys.exit(1)

    # --- Env & Agent Init ---
    env = SumoEnv(config)
    agent = DQNAgent(config)

    # --- Logging & Saving Setup ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = "dqn"
    if config.use_per: exp_name += "_per"
    if config.use_n_step: exp_name += f"_n{config.n_step}"
    if config.use_distributional: exp_name += "_c51"
    if config.use_noisy_nets: exp_name += "_noisy"
    if config.use_double_dqn: exp_name += "_ddqn"
    exp_name += "_revised" # Indicate revised version
    results_dir = f"{exp_name}_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"结果将保存在: {results_dir}")
    print(f"配置: PER={config.use_per}, N-Step={config.n_step if config.use_n_step else 'Off'}, C51={config.use_distributional}, Noisy={config.use_noisy_nets}, DDQN={config.use_double_dqn}")
    print(f"LR={config.initial_learning_rate}, TargetUpdate={config.target_update_freq}, StartLearn={config.learning_starts}, BatchSize={config.batch_size}")
    print(f"奖励参数: FastLaneBonus={config.reward_faster_lane_bonus}, StaySlowPen={config.reward_staying_slow_penalty_scale}, LC_Penalty={config.reward_lane_change_penalty}, SpeedScale={config.reward_high_speed_scale}")
    print(f"归一化: Obs={config.normalize_observations}, Reward={config.normalize_rewards}")
    print(f"训练设备: {agent.device}")

    # Save config used for this run
    try:
        config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        config_save_path = os.path.join(results_dir, "config_used.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
             json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e: print(f"警告: 无法保存配置 JSON: {e}")

    # Metrics lists
    all_rewards_sum_norm = []; lane_change_counts = []; collision_counts = []
    total_steps_per_episode = []; avg_losses_per_episode = []; beta_values = []
    best_avg_reward = -float('inf'); global_step_count = 0
    n_step_buffer = deque(maxlen=config.n_step if config.use_n_step else 1) # Maxlen 1 if n_step off

    # --- DQN Training Loop ---
    print("\n" + "#"*20 + f" 开始 DQN 训练 ({exp_name}) " + "#"*20)
    try:
        for episode in tqdm(range(1, config.dqn_episodes + 1), desc="DQN 训练回合"):
            try:
                state_norm = env.reset() # Get initial normalized state
            except (RuntimeError, ConnectionError) as e_reset:
                 print(f"\n错误: 回合 {episode} 重置环境失败: {e_reset}。跳过此回合。")
                 time.sleep(2) # Wait before next attempt
                 continue # Skip to next episode

            if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                print(f"\n错误: 回合 {episode} 的初始状态无效。尝试再次重置...")
                time.sleep(1)
                try: state_norm = env.reset()
                except (RuntimeError, ConnectionError) as e_reset2:
                    print(f"\n错误: 回合 {episode} 第二次重置环境失败: {e_reset2}。跳过此回合。")
                    time.sleep(2); continue
                if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                    print("\n错误: 重置后初始状态仍然无效。中止训练。")
                    raise RuntimeError("无效的初始状态。")

            episode_reward_norm_sum = 0.0; episode_loss_sum = 0.0; episode_loss_count = 0
            done = False; step_count = 0
            n_step_buffer.clear()

            while not done and step_count < config.max_steps:
                # Use last known raw state's lane index for action masking
                # Make sure last_raw_state is valid
                if env.last_raw_state is not None and not (np.any(np.isnan(env.last_raw_state)) or np.any(np.isinf(env.last_raw_state))):
                    current_lane_idx = int(round(env.last_raw_state[1]))
                    current_lane_idx = np.clip(current_lane_idx, 0, env.num_lanes - 1)
                else:
                    current_lane_idx = env.last_lane_idx # Fallback to last known good lane index

                # Ensure state_norm is valid before passing to agent
                if state_norm is None or np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                    print(f"\n警告: 回合 {episode} 步骤 {step_count} 的状态无效。使用最后一个有效状态。")
                    state_norm = env.last_norm_state.copy() # Use last known good state
                    if state_norm is None or np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                        print("\n错误: 无法恢复有效状态。中止回合。")
                        done = True; break # End episode if state recovery fails

                action = agent.get_action(state_norm, current_lane_idx)
                try:
                    next_state_norm, norm_reward, done, next_lane_idx = env.step(action)
                except traci.exceptions.FatalTraCIError as e_traci_fatal:
                    print(f"\n错误: 回合 {episode} 步骤 {step_count} 发生致命 TraCI 错误: {e_traci_fatal}。中止回合。")
                    done = True; next_state_norm = state_norm; norm_reward = config.reward_collision; next_lane_idx = current_lane_idx # Assume collision
                    env.collision_occurred = True # Mark collision
                except Exception as e_step:
                    print(f"\n错误: 回合 {episode} 步骤 {step_count} env.step 期间发生未知错误: {e_step}")
                    traceback.print_exc()
                    done = True; next_state_norm = state_norm; norm_reward = config.reward_collision; next_lane_idx = current_lane_idx # Assume collision
                    env.collision_occurred = True # Mark collision

                # Basic validation of returned values
                if next_state_norm is None or np.any(np.isnan(next_state_norm)) or np.any(np.isinf(next_state_norm)):
                    # print(f"警告: 回合 {episode} 步骤 {step_count} 从 env.step 返回了无效的 next_state_norm。使用当前状态。") # Reduce spam
                    next_state_norm = state_norm.copy()
                    # Consider ending episode if state becomes invalid consistently
                    # done = True
                if not np.isfinite(norm_reward):
                     # print(f"警告: 回合 {episode} 步骤 {step_count} 从 env.step 返回了无效的 norm_reward ({norm_reward})。设置为 0。") # Reduce spam
                     norm_reward = 0.0


                # N-Step Buffer Logic
                if config.use_n_step:
                    # Store transition in temporary buffer
                    n_step_buffer.append(NStepTransition(state_norm, action, norm_reward, next_state_norm, done, next_lane_idx))

                    # If buffer is full OR episode ended, process the oldest transition
                    if len(n_step_buffer) >= config.n_step or (done and len(n_step_buffer) > 0):
                        n_step_actual = len(n_step_buffer)
                        n_step_return_discounted = 0.0
                        # Calculate N-step discounted return G_t:t+n
                        for i in range(n_step_actual):
                            n_step_return_discounted += (config.gamma**i) * n_step_buffer[i].reward

                        # Get the state/action at time t (oldest in buffer)
                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        # Get the state/done/lane at time t+n (newest in buffer)
                        s_t_plus_n = n_step_buffer[-1].next_state
                        done_n_step = n_step_buffer[-1].done
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index

                        # Create the N-step experience tuple
                        exp = Experience(s_t, a_t, n_step_return_discounted, s_t_plus_n, done_n_step, n_step_actual, next_lane_idx_n_step)

                        # Push the N-step experience to the main replay buffer
                        try:
                            # Ensure states in experience are valid before pushing
                           if exp.state is not None and exp.next_state is not None and \
                              not (np.any(np.isnan(exp.state)) or np.any(np.isinf(exp.state))) and \
                              not (np.any(np.isnan(exp.next_state)) or np.any(np.isinf(exp.next_state))) and \
                              np.isfinite(exp.n_step_reward):

                               if config.use_per: agent.replay_buffer.push(exp)
                               else: agent.replay_buffer.append(exp) # Assumes standard buffer if PER off
                           # else: print(f"警告: 跳过推送无效的 N-step experience 到缓冲区。") # Reduce spam
                        except Exception as e_push:
                            print(f"将 experience 推送到缓冲区时出错: {e_push}"); traceback.print_exc()

                        # Remove the processed transition if buffer was full
                        if len(n_step_buffer) >= config.n_step:
                            n_step_buffer.popleft()
                else: # Standard 1-step DQN
                    exp = Experience(state_norm, action, norm_reward, next_state_norm, done, 1, next_lane_idx)
                    try:
                       if exp.state is not None and exp.next_state is not None and \
                           not (np.any(np.isnan(exp.state)) or np.any(np.isinf(exp.state))) and \
                           not (np.any(np.isnan(exp.next_state)) or np.any(np.isinf(exp.next_state))) and \
                           np.isfinite(exp.n_step_reward): # n_step_reward is just norm_reward here

                           if config.use_per: agent.replay_buffer.push(exp)
                           else: agent.replay_buffer.append(exp)
                       # else: print(f"警告: 跳过推送无效的 1-step experience 到缓冲区。") # Reduce spam
                    except Exception as e_push: print(f"将 experience 推送到缓冲区时出错: {e_push}"); traceback.print_exc()


                # Update state
                state_norm = next_state_norm # Already copied if it was invalid

                episode_reward_norm_sum += norm_reward; step_count += 1; global_step_count += 1

                # Perform DQN Update only AFTER learning_starts steps and if buffer has enough samples
                if global_step_count >= config.learning_starts and len(agent.replay_buffer) >= config.batch_size:
                    loss = agent.update(global_step_count)
                    if loss is not None:
                        if not np.isfinite(loss):
                             print(f"\n警告: 回合 {episode} 更新步骤 {agent.train_step_count} 损失无效 ({loss})。跳过记录。")
                        else:
                             episode_loss_sum += loss; episode_loss_count += 1
                # else:
                    # loss = None # Agent hasn't started learning yet or buffer too small

                if env.collision_occurred: done = True # Ensure done is True if collision flag set by env

                # If episode ended, process any remaining transitions in n_step_buffer
                if done and config.use_n_step and len(n_step_buffer) > 0:
                     while len(n_step_buffer) > 0:
                        n_step_actual = len(n_step_buffer); n_step_return_discounted = 0.0
                        for i in range(n_step_actual): n_step_return_discounted += (config.gamma**i) * n_step_buffer[i].reward
                        s_t = n_step_buffer[0].state; a_t = n_step_buffer[0].action
                        s_t_plus_n = n_step_buffer[-1].next_state
                        # For remaining buffer, done is True for the final transition added
                        done_n_step = True
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index
                        exp = Experience(s_t, a_t, n_step_return_discounted, s_t_plus_n, done_n_step, n_step_actual, next_lane_idx_n_step)
                        try:
                           if exp.state is not None and exp.next_state is not None and \
                              not (np.any(np.isnan(exp.state)) or np.any(np.isinf(exp.state))) and \
                              not (np.any(np.isnan(exp.next_state)) or np.any(np.isinf(exp.next_state))) and \
                              np.isfinite(exp.n_step_reward):

                               if config.use_per: agent.replay_buffer.push(exp)
                               else: agent.replay_buffer.append(exp)
                           # else: print(f"警告: 跳过推送无效的最终 N-step experience 到缓冲区。") # Reduce spam
                        except Exception as e_push: print(f"将最终 experience 推送到缓冲区时出错: {e_push}"); traceback.print_exc()
                        n_step_buffer.popleft()

            # --- Episode End ---
            all_rewards_sum_norm.append(episode_reward_norm_sum)
            lane_change_counts.append(env.change_lane_count)
            collision_counts.append(1 if env.collision_occurred else 0)
            total_steps_per_episode.append(step_count)
            avg_loss_ep = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0
            avg_losses_per_episode.append(avg_loss_ep)

            current_beta = linear_decay(config.per_beta_start, config.per_beta_end, config.per_beta_annealing_steps, global_step_count)
            beta_values.append(current_beta)

            # --- Save Best Model ---
            avg_window = 20
            if episode >= avg_window:
                # Use slicing that handles reaching the end of the list
                rewards_slice = all_rewards_sum_norm[-avg_window:]
                current_avg_reward = np.mean(rewards_slice) if rewards_slice else -float('inf')
                if current_avg_reward > best_avg_reward and global_step_count >= config.learning_starts: # Only save if learning started
                    best_avg_reward = current_avg_reward
                    best_model_path = os.path.join(models_dir, "best_model.pth")
                    torch.save(agent.policy_net.state_dict(), best_model_path)
                    print(f"\n🎉 新的最佳平均奖励 ({avg_window}回合, 归一化总和): {best_avg_reward:.2f}! 模型已保存。")


            # --- Periodic Logging ---
            if episode % config.log_interval == 0:
                 log_slice_start = max(0, episode - config.log_interval)
                 # Use slicing that handles reaching the end of the list
                 avg_reward_log = np.mean(all_rewards_sum_norm[-config.log_interval:]) if all_rewards_sum_norm else 0
                 avg_steps_log = np.mean(total_steps_per_episode[-config.log_interval:]) if total_steps_per_episode else 0
                 collision_rate_log = np.mean(collision_counts[-config.log_interval:]) * 100 if collision_counts else 0
                 # Calculate avg loss only from the last log_interval episodes where learning actually happened
                 valid_losses_log = [l for i, l in enumerate(avg_losses_per_episode) if i >= log_slice_start and total_steps_per_episode[i] > 0 and (global_step_count - sum(total_steps_per_episode[i+1:]) >= config.learning_starts)] # Approx check if learning happened in this ep range
                 avg_loss_log = np.mean([l for l in avg_losses_per_episode[-config.log_interval:] if l > 0]) if any(l > 0 for l in avg_losses_per_episode[-config.log_interval:]) else 0.0

                 avg_lc_log = np.mean(lane_change_counts[-config.log_interval:]) if lane_change_counts else 0
                 buffer_fill = f"{len(agent.replay_buffer)}/{config.replay_buffer_size}" if agent.replay_buffer else "N/A"

                 print(f"\n回合: {episode}/{config.dqn_episodes} | Avg Rew (Norm, {config.log_interval}ep): {avg_reward_log:.2f} "
                       f"| Best Avg ({avg_window}ep): {best_avg_reward:.2f} "
                       f"| Avg Steps: {avg_steps_log:.1f} | Avg LC: {avg_lc_log:.1f} "
                       f"| Coll Rate: {collision_rate_log:.1f}% "
                       f"| Avg Loss: {avg_loss_log:.4f} | Beta: {current_beta:.3f} | Buffer: {buffer_fill} | Steps: {global_step_count}")

            # --- Periodic Model Saving ---
            if episode % config.save_interval == 0:
                 periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                 torch.save(agent.policy_net.state_dict(), periodic_model_path)
                 # print(f"模型已保存: {periodic_model_path}") # Optional log

    except KeyboardInterrupt: print("\n用户中断训练。")
    except Exception as e: print(f"\n训练期间发生致命错误: {e}"); traceback.print_exc()
    finally:
        print("正在关闭最终环境...")
        env._close() # Ensure environment is closed

        if 'agent' in locals() and agent is not None and agent.policy_net is not None:
            last_model_path = os.path.join(models_dir, "last_model.pth")
            torch.save(agent.policy_net.state_dict(), last_model_path)
            print(f"最终模型已保存至: {last_model_path}")

            # --- Plotting (Extended) ---
            print("\n--- 生成训练图表 ---")
            plot_window = 20 # Rolling average window for plots

            # --- Plot 1: Rewards and Steps ---
            try:
                print("正在生成奖励和步数图...")
                fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Increased fig size
                fig1.suptitle("DQN 训练: 奖励和回合步数 (修订版)", fontsize=16)

                # Subplot 1.1: Rolling Average Normalized Reward Sum
                ax = axs1[0]
                ax.set_ylabel(f"{plot_window}回合滚动平均奖励 (归一化)")
                ax.grid(True, linestyle='--')
                if len(all_rewards_sum_norm) >= plot_window:
                    rolling_avg_reward = calculate_rolling_average(all_rewards_sum_norm, plot_window)
                    episode_axis_rolled = np.arange(plot_window - 1, len(all_rewards_sum_norm)) + 1 # Shift axis to match episode numbers
                    ax.plot(episode_axis_rolled, rolling_avg_reward, label=f'{plot_window}回合滚动平均奖励', color='red', linewidth=2)
                    # Plot raw points lightly
                    episode_axis_raw = np.arange(1, len(all_rewards_sum_norm) + 1)
                    ax.plot(episode_axis_raw, all_rewards_sum_norm, label='每回合奖励 (归一化)', color='red', alpha=0.2, marker='.', linestyle='None')
                    ax.legend(loc='best')
                else:
                     ax.text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                # Subplot 1.2: Rolling Average Steps
                ax = axs1[1]
                ax.set_xlabel("回合")
                ax.set_ylabel(f"{plot_window}回合滚动平均步数")
                ax.grid(True, linestyle='--')
                if len(total_steps_per_episode) >= plot_window:
                     rolling_avg_steps = calculate_rolling_average(total_steps_per_episode, plot_window)
                     episode_axis_rolled = np.arange(plot_window - 1, len(total_steps_per_episode)) + 1 # Shift axis
                     ax.plot(episode_axis_rolled, rolling_avg_steps, label=f'{plot_window}回合滚动平均步数', color='blue', linewidth=2)
                     # Plot raw points lightly
                     episode_axis_raw = np.arange(1, len(total_steps_per_episode) + 1)
                     ax.plot(episode_axis_raw, total_steps_per_episode, label='每回合步数', color='blue', alpha=0.2, marker='.', linestyle='None')
                     ax.legend(loc='best')
                else:
                     ax.text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path1 = os.path.join(results_dir, "dqn_training_rewards_steps.png")
                plt.savefig(plot_path1)
                plt.close(fig1)
                print(f"奖励/步数图已保存至: {plot_path1}")
            except Exception as e_plot1: print(f"绘制奖励/步数图时出错: {e_plot1}"); traceback.print_exc()

            # --- Plot 2: Loss ---
            try:
                print("正在生成损失图...")
                fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5)) # Single plot for loss
                fig2.suptitle("DQN 训练: 损失 (修订版)", fontsize=16)
                ax2.set_xlabel("更新步骤")
                ax2.set_ylabel("损失 (交叉熵)")
                ax2.grid(True, linestyle='--')
                loss_per_step = agent.loss_history # Loss per update step
                if loss_per_step: # Check if list is not empty
                     update_axis = np.arange(len(loss_per_step))
                     # Plot raw loss per step with high alpha
                     ax2.plot(update_axis, loss_per_step, label='损失/更新步骤', color='purple', alpha=0.3, linewidth=0.5)
                     # Plot rolling average of loss per step
                     loss_plot_window = max(50, min(1000, len(loss_per_step)//10)) # Dynamic window for loss
                     if len(loss_per_step) >= loss_plot_window and loss_plot_window > 0:
                          rolling_loss = calculate_rolling_average(loss_per_step, loss_plot_window)
                          update_axis_rolled = np.arange(loss_plot_window - 1, len(loss_per_step))
                          ax2.plot(update_axis_rolled, rolling_loss, label=f'{loss_plot_window}步滚动平均损失', color='purple', linewidth=1.5)
                     ax2.legend(loc='best')
                else:
                     # Handle empty loss history case
                     ax2.text(0.5, 0.5, '无损失数据（训练未开始？）', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path2 = os.path.join(results_dir, "dqn_training_loss.png")
                plt.savefig(plot_path2)
                plt.close(fig2)
                print(f"损失图已保存至: {plot_path2}")
            except Exception as e_plot2: print(f"绘制损失图时出错: {e_plot2}"); traceback.print_exc()


            # --- Plot 3: PER Beta ---
            if config.use_per:
                try:
                    print("正在生成 PER Beta 图...")
                    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
                    fig3.suptitle("DQN 训练: PER Beta 退火 (修订版)", fontsize=16)
                    ax3.set_xlabel("回合")
                    ax3.set_ylabel("Beta 值")
                    ax3.grid(True, linestyle='--')
                    if beta_values:
                         episode_axis_beta = np.arange(1, len(beta_values) + 1) # Start episode axis from 1
                         ax3.plot(episode_axis_beta, beta_values, label='PER Beta', color='green')
                         ax3.legend()
                    else:
                        ax3.text(0.5, 0.5, '无 Beta 数据', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plot_path3 = os.path.join(results_dir, "dqn_training_beta.png")
                    plt.savefig(plot_path3)
                    plt.close(fig3)
                    print(f"Beta 图已保存至: {plot_path3}")
                except Exception as e_plot3: print(f"绘制 Beta 图时出错: {e_plot3}"); traceback.print_exc()


            # --- Plot 4: Collisions and Lane Changes ---
            try:
                print("正在生成碰撞和换道图...")
                fig4, axs4 = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Increased fig size
                fig4.suptitle("DQN 训练: 碰撞率和换道次数 (修订版)", fontsize=16)

                # Subplot 4.1: Collision Rate
                ax = axs4[0]
                ax.set_ylabel(f"{plot_window}回合滚动平均碰撞率 (%)")
                ax.grid(True, linestyle='--')
                ax.set_ylim(-5, 105) # Set Y axis from 0 to 100%
                if len(collision_counts) >= plot_window:
                    collision_rate = np.array(collision_counts) * 100
                    rolling_avg_coll = calculate_rolling_average(collision_rate, plot_window)
                    episode_axis_rolled_coll = np.arange(plot_window - 1, len(collision_counts)) + 1 # Shift axis
                    ax.plot(episode_axis_rolled_coll, rolling_avg_coll, label=f'{plot_window}回合滚动平均碰撞率', color='black', linewidth=2)
                    # Plot raw points lightly
                    episode_axis_raw = np.arange(1, len(collision_counts) + 1)
                    ax.plot(episode_axis_raw, collision_rate, label='每回合碰撞 (0/100)', color='black', alpha=0.2, marker='.', linestyle='None')
                    ax.legend(loc='best')
                else:
                    ax.text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                # Subplot 4.2: Lane Changes
                ax = axs4[1]
                ax.set_xlabel("回合")
                ax.set_ylabel(f"{plot_window}回合滚动平均换道次数")
                ax.grid(True, linestyle='--')
                if len(lane_change_counts) >= plot_window:
                    rolling_avg_lc = calculate_rolling_average(lane_change_counts, plot_window)
                    episode_axis_rolled_lc = np.arange(plot_window - 1, len(lane_change_counts)) + 1 # Shift axis
                    ax.plot(episode_axis_rolled_lc, rolling_avg_lc, label=f'{plot_window}回合滚动平均换道次数', color='orange', linewidth=2)
                    # Plot raw points lightly
                    episode_axis_raw = np.arange(1, len(lane_change_counts) + 1)
                    ax.plot(episode_axis_raw, lane_change_counts, label='每回合换道次数', color='orange', alpha=0.2, marker='.', linestyle='None')
                    ax.legend(loc='best')
                else:
                    ax.text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path4 = os.path.join(results_dir, "dqn_training_collisions_lc.png")
                plt.savefig(plot_path4)
                plt.close(fig4)
                print(f"碰撞/换道图已保存至: {plot_path4}")
            except Exception as e_plot4: print(f"绘制碰撞/换道图时出错: {e_plot4}"); traceback.print_exc()


            # --- Save Training Data ---
            print("正在保存训练数据...")
            training_data = {
                "episode_rewards_sum_norm": all_rewards_sum_norm,
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "avg_losses_per_episode": avg_losses_per_episode, # Avg loss per episode
                "detailed_loss_history_per_step": agent.loss_history, # Loss per update step
                "beta_values_per_episode": beta_values
            }
            data_path = os.path.join(results_dir, "training_data_final_revised.json")
            try:
                # Custom serializer function (FIXED: no 'self')
                def default_serializer(obj):
                    if isinstance(obj, (np.integer, np.int_)): return int(obj)
                    elif isinstance(obj, (np.floating, np.float_)): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, np.bool_): return bool(obj)
                    elif isinstance(obj, (datetime.datetime, datetime.date)): return obj.isoformat()
                    # Fallback for types JSON doesn't handle directly but are safe
                    elif isinstance(obj, (int, float, bool, str, list, dict, type(None))):
                         return obj
                    # If none of the above match, convert to string as a final fallback
                    return str(obj)

                # Clean data before saving (handle potential non-finite numbers)
                cleaned_training_data = {}
                for key, value_list in training_data.items():
                    if isinstance(value_list, list):
                        cleaned_list = []
                        for item in value_list:
                            # Replace non-finite floats/numpy floats with 0.0
                            if isinstance(item, (float, np.floating)) and not np.isfinite(item):
                                cleaned_list.append(0.0)
                            else:
                                # Apply serializer to ensure everything is JSON compatible
                                cleaned_list.append(default_serializer(item))
                        cleaned_training_data[key] = cleaned_list
                    else: # Should not happen based on current structure, but good practice
                        cleaned_training_data[key] = default_serializer(value_list)


                with open(data_path, "w", encoding="utf-8") as f:
                     json.dump(cleaned_training_data, f, indent=4, ensure_ascii=False) # Use cleaned data
                print(f"训练数据已保存至: {data_path}")
            except TypeError as e_type:
                 print(f"保存训练数据时发生类型错误: {e_type}")
                 # Optionally print problematic data structure
                 import pprint
                 print("Problematic data structure snippet:")
                 pprint.pprint({k: v[:10] if isinstance(v, list) else v for k, v in cleaned_training_data.items()}) # Print first 10 items
            except Exception as e:
                 print(f"保存训练数据时发生未知错误: {e}"); traceback.print_exc()
        else: print("智能体未初始化或无策略网络，无法保存最终模型/数据。")
        print(f"\n DQN 训练 ({exp_name}) 完成。结果已保存在: {results_dir}")

if __name__ == "__main__":
    main()