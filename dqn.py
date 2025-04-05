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
    dqn_episodes = 500 # 与 PPO 保持相同的回合数以便比较
    max_steps = 8000   # 与 PPO 保持相同的每回合最大步数
    log_interval = 10
    save_interval = 50

    # --- DQN 超参数 ---
    gamma = 0.99 # 折扣因子 (与 PPO 相同)
    initial_learning_rate = 7e-5 # C51/Noisy 可能需要更低的学习率 (原为 1e-4)
    batch_size = 512 # 从回放缓冲区采样的批量大小 (与 PPO 相同)
    hidden_size = 256 # 网络隐藏层大小 (与 PPO 相同)
    replay_buffer_size = 100000 # 经验回放缓冲区大小
    target_update_freq = 2500   # 目标网络更新的训练 *步数* 间隔 (原为 1000)
    learning_starts = 7500      # 开始训练前收集的步数 (原为 5000)
    use_double_dqn = True       # 使用 Double DQN 改进 (在 C51 逻辑内应用)

    # --- Epsilon-Greedy Exploration (REMOVED - Using Noisy Nets) ---
    # epsilon_start = 1.0
    # epsilon_end = 0.05
    # epsilon_decay_total_steps = int(dqn_episodes * 300 * 0.7)

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
    # 保留原始估计，可能需要根据实际的步数/回合进行调整
    per_beta_annealing_steps = int(dqn_episodes * 300 * 0.8) # 在估计总步数的 80% 内退火
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
        batch_count = x.shape[0]
        # EMA 更新
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1.0 - self.alpha) * self.var + self.alpha * batch_var
        self.count += batch_count # 保持计数以供参考，但 EMA 驱动更新

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8) # 为安全起见，此处也添加 epsilon

# 奖励缩放助手 - 现在归一化单步奖励
class RewardNormalizer:
    def __init__(self, gamma: float, epsilon: float = 1e-8, alpha: float = 0.001, clip: float = 10.0):
        self.ret_rms = RunningMeanStd(shape=(), alpha=alpha) # 使用 RunningMeanStd 处理回报
        self.epsilon = epsilon
        self.clip = clip
        # Gamma 不再直接在此处使用，但为了签名一致性而保留

    def normalize(self, r: np.ndarray) -> np.ndarray:
        # 使用当前奖励更新运行统计信息
        self.ret_rms.update(r) # 假设 r 是单个奖励或一批奖励
        # 归一化: (r - mean) / std, 但通常奖励只使用 r / std
        norm_r = r / self.ret_rms.std
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
    # print("尝试终止残留的 SUMO 进程...") # 减少冗余信息
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
    time.sleep(0.1) # 较短的睡眠时间

# 线性衰减函数 (用于 PER beta)
def linear_decay(start_val, end_val, total_steps, current_step):
    """线性衰减计算"""
    if current_step >= total_steps:
        return end_val
    return start_val + (end_val - start_val) * (current_step / total_steps)

#####################
#   SUMO 环境封装    #
#####################
# 为观测值归一化存储和奖励计算调整稍作修改
class SumoEnv:
    def __init__(self, config: Config):
        self.config = config
        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_speed = 0.0 # 用于舒适度惩罚
        self.last_raw_state = np.zeros(config.state_dim) # 存储归一化前的原始状态
        self.last_norm_state = np.zeros(config.state_dim) # 存储归一化后的状态
        self.last_lane_idx = 0 # 存储上一个车道索引用于奖励计算

        # 归一化 (在环境内部应用)
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate, clip=config.reward_norm_clip) if config.normalize_rewards else None

        # 指标
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
            "--collision.action", "warn", # 使用 "warn" 检测碰撞而不立即停止仿真
            "--time-to-teleport", "-1", # 禁用传送
            "--no-warnings", "true",
            "--seed", str(np.random.randint(0, 10000))
        ]
        if self.config.sumo_binary == "sumo-gui":
             sumo_cmd.extend(["--quit-on-end", "true"]) # 自动关闭 GUI

        try:
             # 仅在非 GUI 模式下重定向 stdout/stderr，以避免抑制 GUI 错误
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
                time.sleep(1.0 + attempt * 0.5) # 略微调整延迟
                traci.init(self.traci_port)
                # print(f"✅ SUMO TraCI 已连接 (端口: {self.traci_port}).") # 减少冗余信息
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
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0") # 红色
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0") # 使其策略灵活
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcSpeedGain", "2.0") # 更愿意为速度而改变 (从 1.0 增加)
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcCooperative", "0.5") # 略微减少合作以优先考虑自身利益
                traci.vehicletype.setParameter(self.config.ego_type_id, "jmIgnoreFoeProb", "0.1") # 如果被阻塞，在合并/换道时有小概率忽略前车
                traci.vehicletype.setParameter(self.config.ego_type_id, "carFollowModel", "IDM") # 使用标准 IDM 模型
                traci.vehicletype.setParameter(self.config.ego_type_id, "minGap", "2.5") # 标准最小间隙
            except traci.exceptions.TraCIException as e: print(f"警告: 设置 Ego 类型 '{self.config.ego_type_id}' 参数失败: {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.config.ego_vehicle_id)
                time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"警告: 移除残留 Ego 失败: {e}")

        try:
            # 首先尝试在特定车道上添加，如果失败则回退到随机
            start_lane = random.choice([0, 1, 2])
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                              typeID=self.config.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max") # 尝试特定车道

            wait_steps = int(2.0 / self.config.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                     ego_appeared = True
                     break
            if not ego_appeared:
                 # 如果特定车道失败 (例如，被阻塞)，则回退
                 print(f"警告: 在车道 {start_lane} 上添加 Ego 失败，尝试随机车道。")
                 traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                                   typeID=self.config.ego_type_id, depart="now",
                                   departLane="random", departSpeed="max")
                 for _ in range(wait_steps):
                     traci.simulationStep()
                     if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                         ego_appeared = True
                         break
                 if not ego_appeared:
                    raise RuntimeError(f"Ego 车辆在 {wait_steps*2} 步内未出现。")

        except traci.exceptions.TraCIException as e:
            print(f"错误: 添加 Ego 车辆 '{self.config.ego_vehicle_id}' 失败: {e}")
            raise RuntimeError("添加 Ego 车辆失败。")


    def reset(self) -> np.ndarray:
        """为新回合重置环境，返回归一化状态"""
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.reset_metrics()

        self.last_speed = 0.0
        self.last_raw_state = np.zeros(self.config.state_dim)
        self.last_norm_state = np.zeros(self.config.state_dim)
        self.last_lane_idx = 0 # 重置上一个车道
        norm_state = np.zeros(self.config.state_dim)

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
             try:
                 self.last_speed = traci.vehicle.getSpeed(self.config.ego_vehicle_id)
                 self.last_lane_idx = traci.vehicle.getLaneIndex(self.config.ego_vehicle_id)
                 # 首先获取原始状态
                 raw_state = self._get_raw_state()
                 self.last_raw_state = raw_state.copy()
                 # 然后进行归一化
                 norm_state = self._normalize_state(raw_state)
                 self.last_norm_state = norm_state.copy()
             except traci.exceptions.TraCIException:
                 print("警告: 在 reset 中的初始状态获取期间发生 TraCI 异常。")
                 pass # 如果出错则保持默认值
        else:
             print("警告: 在 reset 中的 add/wait 后未立即找到 Ego 车辆。")

        return norm_state # 返回归一化状态

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """使用运行均值/标准差归一化状态。"""
        if self.obs_normalizer:
            self.obs_normalizer.update(state.reshape(1, -1)) # 使用单个观测值更新
            norm_state = (state - self.obs_normalizer.mean) / self.obs_normalizer.std
            norm_state = np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip)
            return norm_state.astype(np.float32)
        else:
            return state # 如果归一化关闭，则返回原始状态


    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """获取最近车辆的距离和相对速度 (未更改)"""
        max_dist = self.config.max_distance
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        ego_road = traci.vehicle.getRoadID(ego_id)
        if ego_road == "" or not ego_road.startswith("E"): return infos # 检查是否在预期的路段上

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                # 首先检查车辆是否在同一路段上
                veh_road = traci.vehicle.getRoadID(veh_id)
                if veh_road != ego_road: continue

                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                dx = veh_pos[0] - ego_pos[0] # 假设主要沿 X 轴行驶
                dy = veh_pos[1] - ego_pos[1] # 横向距离
                distance = math.sqrt(dx**2 + dy**2) # 欧氏距离

                if distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed # 如果 ego 更快则为正

                # 更仔细地检查车道相对位置
                if veh_lane == ego_lane: # 同一车道
                    if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                    # 对于此状态表示，忽略同一车道后面的车辆
                elif veh_lane == ego_lane - 1: # 左车道
                    if dx > -5 and distance < infos['left_front'][0]: # 略微靠后到前方
                        infos['left_front'] = (distance, rel_speed)
                    elif dx <= -5 and distance < infos['left_back'][0]: # 更靠后
                        infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1: # 右车道
                     if dx > -5 and distance < infos['right_front'][0]: # 略微靠后到前方
                        infos['right_front'] = (distance, rel_speed)
                     elif dx <= -5 and distance < infos['right_back'][0]: # 更靠后
                        infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue # 如果发生 TraCI 错误，则跳过该车辆
        return infos

    def _get_raw_state(self) -> np.ndarray:
        """获取当前环境状态 (归一化之前的原始值)"""
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            return self.last_raw_state # 如果 ego 消失，则返回最后已知的原始状态

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if current_road == "" or not current_road.startswith("E"):
                return self.last_raw_state # 不在预期的路段上

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            num_lanes = traci.edge.getLaneNumber("E0") # 获取实际车道数

            # 在使用 ego_lane 进行检查之前，确保其有效
            if not (0 <= ego_lane < num_lanes):
                 # 尝试重新获取或使用最后已知的有效车道
                 print(f"警告: 检测到无效的 ego 车道 {ego_lane}。尝试重新获取...")
                 time.sleep(0.05)
                 if ego_id in traci.vehicle.getIDList():
                     ego_lane = traci.vehicle.getLaneIndex(ego_id)
                 if not (0 <= ego_lane < num_lanes):
                     print(f"警告: 仍然是无效车道 {ego_lane}。使用最后有效车道 {self.last_lane_idx}。")
                     ego_lane = self.last_lane_idx
                     if not (0 <= ego_lane < num_lanes): # 最后的回退
                         ego_lane = 0

            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # --- 原始状态特征 ---
            state[0] = ego_speed
            state[1] = float(ego_lane) # 保持车道索引为 float 以实现一致性/归一化
            state[2] = min(surround_info['front'][0], self.config.max_distance)
            state[3] = surround_info['front'][1] # 相对速度 (此处不裁剪)
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
                return self.last_raw_state

            self.last_raw_state = state.copy() # 存储最新的有效原始状态

        except traci.exceptions.TraCIException as e:
            # 检查是否是 "vehicle not found" 错误，这在接近结束/碰撞时可能是预期的
            if "Vehicle '" + ego_id + "' is not known" in str(e):
                 pass # 如果车辆离开仿真或碰撞，则为预期
            else:
                 print(f"警告: 获取 {ego_id} 的原始状态时发生 TraCI 错误: {e}。返回最后已知的原始状态。")
            return self.last_raw_state
        except Exception as e:
            print(f"警告: 获取 {ego_id} 的原始状态时发生未知错误: {e}。返回最后已知的原始状态。")
            traceback.print_exc()
            return self.last_raw_state

        return state # 返回原始状态


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """执行一个动作，返回 (next_normalized_state, normalized_reward, done, next_lane_index)"""
        done = False
        raw_reward = 0.0 # 归一化之前的奖励
        next_lane_index = self.last_lane_idx # 使用当前车道初始化
        ego_id = self.config.ego_vehicle_id
        self.last_action = action # 存储采取的动作

        if ego_id not in traci.vehicle.getIDList():
             # 返回最后已知的归一化状态，碰撞奖励，done=True，最后已知的车道索引
             return self.last_norm_state, self.reward_normalizer.normalize(np.array([self.config.reward_collision]))[0] if self.reward_normalizer else self.config.reward_collision, True, self.last_lane_idx

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id)
            current_road = traci.vehicle.getRoadID(ego_id)
            num_lanes = traci.edge.getLaneNumber(current_road) if current_road.startswith("E") else 0

             # 在执行动作之前确保 current_lane 有效
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)


            # 1. 如果有效则执行动作
            lane_change_requested = False
            if action == 1 and current_lane > 0: # 尝试左转
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0) # 请求换道
                lane_change_requested = True
            elif action == 2 and num_lanes > 0 and current_lane < (num_lanes - 1): # 尝试右转
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0) # 请求换道
                lane_change_requested = True

            # 2. 仿真步骤
            traci.simulationStep()
            self.current_step += 1

            # 3. 在步骤之后检查状态并计算奖励
            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True
                done = True
                raw_reward = self.config.reward_collision
                next_norm_state = self.last_norm_state # 返回最后已知的归一化状态
                next_lane_index = self.last_lane_idx # 使用最后已知的车道
                normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward
                return next_norm_state, normalized_reward, done, next_lane_index

            # 显式碰撞检查
            collisions = traci.simulation.getCollisions()
            ego_collided = False
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True
                    ego_collided = True
                    done = True
                    raw_reward = self.config.reward_collision
                    break

            # 获取下一个状态 (先原始，然后归一化)
            next_raw_state = self._get_raw_state()
            next_norm_state = self._normalize_state(next_raw_state)
            next_lane_index = int(round(next_raw_state[1])) # 获取步骤 *之后* 的车道索引

            # 除非发生碰撞，否则根据步骤 *之后* 的状态计算奖励
            if not self.collision_occurred:
                 # 从 *原始* 状态中提取必要信息以计算奖励
                 current_speed_after_step = next_raw_state[0]
                 current_lane_after_step = int(round(next_raw_state[1])) # = next_lane_index
                 front_dist_after_step = next_raw_state[2]
                 left_front_dist = next_raw_state[4]
                 right_front_dist = next_raw_state[7]
                 can_change_left_after_step = next_raw_state[10] > 0.5
                 can_change_right_after_step = next_raw_state[11] > 0.5


                 # 检查是否实际发生了换道
                 actual_lane_change = (self.last_lane_idx != current_lane_after_step)
                 if actual_lane_change:
                     self.change_lane_count += 1
                 # 仅当此步骤中 *请求* 了换道时才应用惩罚
                 effective_action = action if lane_change_requested else 0

                 # 将必要的状态组件传递给奖励函数
                 raw_reward = self._calculate_reward(effective_action, current_speed_after_step,
                                                 current_lane_after_step, front_dist_after_step,
                                                 self.last_lane_idx, # 传递前一个车道
                                                 can_change_left_after_step, can_change_right_after_step,
                                                 left_front_dist, right_front_dist)

            # 4. 检查其他终止条件
            if traci.simulation.getTime() >= 3600: done = True
            if self.current_step >= self.config.max_steps: done = True

        except traci.exceptions.TraCIException as e:
            if ego_id not in traci.vehicle.getIDList(): self.collision_occurred = True
            raw_reward = self.config.reward_collision # 因错误/碰撞而惩罚
            done = True
            next_norm_state = self.last_norm_state
            next_lane_index = self.last_lane_idx
        except Exception as e:
            print(f"错误: 在步骤 {self.current_step} 期间发生未知异常: {e}")
            traceback.print_exc()
            done = True
            raw_reward = self.config.reward_collision
            self.collision_occurred = True # 在未知错误时假设发生碰撞
            next_norm_state = self.last_norm_state
            next_lane_index = self.last_lane_idx

        # 5. 更新下一个步骤的最后状态信息
        self.last_speed = next_raw_state[0]
        self.last_lane_idx = next_lane_index # 根据步骤 *之后* 的状态更新最后车道索引
        self.last_norm_state = next_norm_state.copy() # 存储归一化状态

        # *** 返回之前归一化奖励 ***
        normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward

        # 返回归一化状态、归一化奖励、完成标志和下一个车道索引
        return next_norm_state, normalized_reward, done, next_lane_index


    def _calculate_reward(self, action_taken: int, current_speed: float, current_lane: int,
                           front_dist: float, previous_lane: int,
                           can_change_left: bool, can_change_right: bool,
                           left_front_dist: float, right_front_dist: float) -> float:
        """根据步骤 *之后* 的状态计算奖励 (使用原始值)，并包含新的惩罚/奖励。"""
        if self.collision_occurred: return 0.0 # 奖励已处理

        try:
            num_lanes = len(self.config.lane_max_speeds)
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)
            if not (0 <= previous_lane < num_lanes): # 确保前一个车道也有效
                 previous_lane = current_lane # 如果无效，默认为当前车道

            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            # --- 奖励组件 ---
            # 速度奖励/惩罚 (增加比例)
            speed_diff = abs(current_speed - target_speed)
            speed_reward = np.exp(- (speed_diff / (target_speed * 0.3 + 1e-6))**2 ) * self.config.reward_high_speed_scale

            # 低速惩罚
            low_speed_penalty = 0.0
            low_speed_threshold = target_speed * 0.6
            if current_speed < low_speed_threshold:
                low_speed_penalty = (current_speed / low_speed_threshold - 1.0) * self.config.reward_low_speed_penalty_scale

            # 换道惩罚和奖励 (减少惩罚，增加奖励)
            lane_change_penalty = 0.0
            lane_change_bonus = 0.0
            if action_taken != 0: # 如果尝试了换道
                lane_change_penalty = self.config.reward_lane_change_penalty
                # 检查换道是否导致移动到潜在更快的车道
                # 假设较低的索引 = 更快的车道 (根据 config.lane_max_speeds)
                if current_lane < previous_lane: # 向左移动到潜在更快的车道
                     lane_change_bonus = self.config.reward_faster_lane_bonus
                # 可以为不必要的向右移动添加惩罚，但保持简单

            # *** 新增: 停留在较慢车道的惩罚 ***
            staying_slow_penalty = 0.0
            if action_taken == 0: # 仅当智能体选择 '保持车道' 时惩罚
                # 检查左侧车道 (更快)
                left_lane_idx = current_lane - 1
                if left_lane_idx >= 0 and self.config.lane_max_speeds[left_lane_idx] > lane_max_speed:
                    if can_change_left and left_front_dist > self.config.min_safe_change_dist:
                        # 左侧有更快的车道可用，可以换道，并且前方足够清晰
                        staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale
                # 检查右侧车道 (更快 - 不太常见的情况，但如果配置错误则可能)
                # 仅当未因左侧选项而被惩罚时才应用惩罚
                elif staying_slow_penalty == 0.0:
                    right_lane_idx = current_lane + 1
                    if right_lane_idx < num_lanes and self.config.lane_max_speeds[right_lane_idx] > lane_max_speed:
                         if can_change_right and right_front_dist > self.config.min_safe_change_dist:
                             # 右侧有更快的车道可用，可以换道，并且前方足够清晰
                             staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale


            # 安全距离惩罚
            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < self.config.max_distance:
                if front_dist < min_safe_dist:
                     safety_dist_penalty = max(-1.0, (front_dist / min_safe_dist - 1.0)) * self.config.safe_distance_penalty_scale

            # 舒适度惩罚
            comfort_penalty = 0.0
            acceleration = (current_speed - self.last_speed) / self.config.step_length
            harsh_braking_threshold = -3.0
            if acceleration < harsh_braking_threshold:
                 comfort_penalty = (acceleration - harsh_braking_threshold) * self.config.reward_comfort_penalty_scale

            # 时间存活奖励
            time_alive = self.config.time_alive_reward

            # --- 总奖励 ---
            total_reward = (speed_reward +
                            low_speed_penalty +
                            lane_change_penalty +
                            lane_change_bonus +
                            staying_slow_penalty + # 添加新惩罚
                            safety_dist_penalty +
                            comfort_penalty +
                            time_alive)

            return total_reward

        except Exception as e:
            print(f"警告: 计算奖励时出错: {e}。返回 0。")
            traceback.print_exc()
            return 0.0

    def _close(self):
        """关闭 SUMO 实例和 TraCI 连接 (与 ppo.py 完全相同)"""
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
                except Exception as e: print(f"警告: SUMO 终止期间出错: {e}")
                self.sumo_process = None
                self.traci_port = None
                time.sleep(0.1)
        else:
            self.traci_port = None


#####################
#   DQN 组件         #
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
        # 鲁棒性检查: 返回前确保数据有效
        data = self.data[dataIdx]
        if not isinstance(data, Experience):
            print(f"警告: 在 SumTree 的 dataIdx {dataIdx} (tree idx {idx}) 处发现损坏的数据。 data: {data}。尝试返回虚拟数据或邻近数据。")
            # 选项 1: 返回虚拟数据 (可能导致下游问题)
            # return (idx, self.tree[idx], Experience(np.zeros(self.config.state_dim), 0, 0, np.zeros(self.config.state_dim), True, 1, 0))
            # 选项 2: 尝试获取邻居 (不太理想)
            # 尝试使用略微不同的 's' 再次采样 - 可能很复杂
            # 最简单的恢复方法: 返回优先级，但在上游指示数据错误
            # 目前，让我们返回潜在的错误数据，让采样循环处理重试
            pass # 允许调用函数 (`sample`) 检测无效类型
        return (idx, self.tree[idx], data)
    def __len__(self): return self.n_entries

# --- Prioritized Replay Buffer (PER + N-Step) --- (逻辑未更改，依赖于 Experience 元组)
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, config: Config):
        self.config = config # 存储配置以供潜在使用 (例如，检查中的 state_dim)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
    def push(self, experience: Experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    def sample(self, batch_size: int, beta: float) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        experiences = []; indices = np.empty((batch_size,), dtype=np.int32); is_weights = np.empty((batch_size,), dtype=np.float32)
        segment = self.tree.total() / batch_size
        if self.tree.total() == 0: # 如果缓冲区为空/损坏，避免除以零
             raise RuntimeError("SumTree 总和为零，无法采样。")

        for i in range(batch_size):
            attempt = 0
            while attempt < 5: # 如果遇到无效数据，重试采样几次
                a = segment * i; b = segment * (i + 1); s = random.uniform(a, b)
                # 确保 s 不超过总和 (在接近末尾时可能由于浮点问题而发生)
                s = min(s, self.tree.total() - 1e-6)
                try:
                    (idx, p, data) = self.tree.get(s)

                    # 关键检查: 确保 'data' 是正确的类型
                    if not isinstance(data, Experience):
                        print(f"警告: 在索引 {idx} 处采样到无效数据 (类型: {type(data)}), tree 总和 {self.tree.total()}, segment [{a:.4f},{b:.4f}], s {s:.4f}。重试采样 ({attempt+1}/5)...")
                        attempt += 1
                        time.sleep(0.01) # 重试前稍作延迟
                        segment = self.tree.total() / batch_size # 重新计算 segment 以防 total 更改
                        continue # 重试循环

                    # 如果需要，检查 Experience 内部的 None 或其他无效内容
                    # 例如，if data.state is None: ... continue

                    experiences.append(data); indices[i] = idx; prob = p / self.tree.total()
                    if prob <= 0: # 处理零概率情况
                         print(f"警告: 采样优先级 {p} 导致零概率 (total: {self.tree.total()})。设置小概率。")
                         prob = self.config.per_epsilon / self.tree.total() # 使用 epsilon

                    is_weights[i] = np.power(self.tree.n_entries * prob, -beta)
                    break # 采样成功，退出重试循环
                except Exception as e:
                    print(f"为 s={s}, idx={idx} 执行 SumTree.get 或处理时出错: {e}")
                    attempt += 1
                    time.sleep(0.01)
                    segment = self.tree.total() / batch_size # 重新计算 segment

            if attempt == 5:
                 raise RuntimeError(f"尝试 5 次后未能从 SumTree 采样到有效的 Experience 数据。 Tree total: {self.tree.total()}, n_entries: {self.tree.n_entries}")


        is_weights /= (is_weights.max() + 1e-8) # 归一化，为稳定性添加 epsilon
        return experiences, is_weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + self.config.per_epsilon # 使用绝对 TD 误差，添加 epsilon
        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < self.tree.capacity * 2 - 1):
                print(f"警告: 提供给 update_priorities 的索引 {idx} 无效。跳过。")
                continue
            if not np.isfinite(priority):
                 print(f"警告: 索引 {idx} 的优先级 {priority} 非有限。进行裁剪。")
                 priority = self.max_priority # 裁剪到已知的最大优先级

            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    def __len__(self) -> int: return self.tree.n_entries

# --- Noisy Linear Layer --- (未更改)
class NoisyLinear(nn.Module):
    """用于分解高斯噪声的 Noisy Linear 层"""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # 权重均值和标准差的可学习参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        # 偏置均值和标准差的可学习参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # 分解噪声参数 (不可学习的缓冲区)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """初始化权重和偏置"""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features)) # 已修正: 偏置 sigma 初始化使用 out_features

    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成缩放后的噪声"""
        x = torch.randn(size, device=self.weight_mu.device) # 确保噪声在正确的设备上
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """生成新的噪声向量"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # 外积创建分解噪声矩阵
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out) # 偏置噪声就是输出噪声

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带噪声权重和偏置的前向传播"""
        if self.training: # 仅在训练期间应用噪声
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else: # 在评估期间使用均值权重/偏置
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


# --- Q-Network (为 C51 和 Noisy Nets 修改) --- (网络结构未更改)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, config: Config):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = config.num_atoms
        self.use_noisy = config.use_noisy_nets
        self.sigma_init = config.noisy_sigma_init # 传递 sigma_init

        # 基于配置的层构造函数
        # 如果使用 NoisyLinear 则传递 sigma_init
        linear = lambda in_f, out_f: NoisyLinear(in_f, out_f, self.sigma_init) if self.use_noisy else nn.Linear(in_f, out_f)

        self.feature_layer = nn.Sequential(
            linear(state_dim, hidden_size), nn.ReLU(),
        )
        # 即使只有一个层使用它，也按照惯例使用 NoisyLinear
        self.value_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.num_atoms) # 输出 V(s) 分布 logits
        )
        self.advantage_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.action_dim * self.num_atoms) # 输出 A(s,a) 分布 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.feature_layer(x)

        values = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantages = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)

        # 结合价值流和优势流 (Dueling 架构)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_logits = values + advantages - advantages.mean(dim=1, keepdim=True)

        # 应用 Softmax 获取每个动作的概率分布
        q_probs = F.softmax(q_logits, dim=2) # 在原子维度上进行 Softmax

        # 避免数值不稳定 (确保概率和为 1 且 > 0)
        q_probs = (q_probs + 1e-8) / (1.0 + self.num_atoms * 1e-8) # 温和地归一化

        return q_probs # 返回分布 [batch, action, atoms]

    def reset_noise(self):
        """重置所有 NoisyLinear 层中的噪声"""
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
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms) # 分布支撑 (原子)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1) # 原子之间的距离

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.support = self.support.to(self.device) # 将支撑移到设备

        # 网络 (现在具有 C51/Noisy 功能)
        self.policy_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 目标网络仅用于推理

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.initial_learning_rate, eps=1e-5) # AdamW 对 NoisyNets 可能稍好

        # 回放缓冲区 (优先)
        self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size, config.per_alpha, config) if config.use_per else None
        if not config.use_per:
             print("警告: PER 已禁用，使用标准回放缓冲区。")
             # 如果 PER 关闭，确保标准缓冲区使用正确的 Experience 元组
             self.replay_buffer = deque([], maxlen=config.replay_buffer_size) # 用于 N-step 元组的简单 deque

        # 归一化助手现在位于 SumoEnv 中

        # 训练步数计数器 (用于目标网络更新)
        self.train_step_count = 0
        self.loss_history = [] # 跟踪损失以进行日志记录

    def get_action(self, normalized_state: np.ndarray, current_lane_idx: int) -> int:
        """根据 (噪声) 策略网络的预期 Q 值选择动作"""
        # 如果使用 Noisy Nets，在动作选择前重置噪声
        if self.config.use_noisy_nets:
            self.policy_net.reset_noise()
            # 这里无需重置 target_net 噪声，因为它不用于动作选择

        self.policy_net.eval() # 设置为评估模式进行推理 (对 Noisy Nets 评估很重要)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            # 获取动作概率分布 [1, action_dim, num_atoms]
            action_probs = self.policy_net(state_tensor)
            # 计算预期 Q 值: 对每个动作求和(概率 * 支撑值)
            expected_q_values = (action_probs * self.support).sum(dim=2) # [1, action_dim]

            # 动作屏蔽: 使不可能的动作的 Q 值无效
            num_lanes = len(self.config.lane_max_speeds)
            q_values_masked = expected_q_values.clone()
            if current_lane_idx == 0:
                q_values_masked[0, 1] = -float('inf') # 不能向左转
            if current_lane_idx >= num_lanes - 1:
                q_values_masked[0, 2] = -float('inf') # 不能向右转

            action = q_values_masked.argmax().item() # 选择具有最高有效预期 Q 值的动作

        self.policy_net.train() # 设置回训练模式
        return action

    def update(self, global_step: int) -> Optional[float]:
        """使用来自回放缓冲区的一批数据更新 Q 网络 (PER + N-Step + C51 + Noisy 感知 + 修复的 DoubleDQN 屏蔽)"""
        if len(self.replay_buffer) < self.config.learning_starts: return None
        if len(self.replay_buffer) < self.config.batch_size: return None

        current_beta = linear_decay(self.config.per_beta_start, self.config.per_beta_end,
                                    self.config.per_beta_annealing_steps, global_step)

        # 从缓冲区采样
        try:
            if self.config.use_per:
                experiences, is_weights, indices = self.replay_buffer.sample(self.config.batch_size, current_beta)
            else: # 如果 PER 关闭，则均匀采样
                experiences = random.sample(self.replay_buffer, self.config.batch_size)
                is_weights = np.ones(self.config.batch_size) # 不需要 IS 权重
                indices = None # 不需要索引
        except RuntimeError as e:
             print(f"从回放缓冲区采样时出错: {e}。跳过更新。")
             return None


        batch = Experience(*zip(*experiences))

        states_np = np.array(batch.state)
        actions_np = np.array(batch.action)
        n_step_rewards_np = np.array(batch.n_step_reward) # 奖励已由 env.step 归一化
        next_states_np = np.array(batch.next_state)
        dones_np = np.array(batch.done)
        n_np = np.array(batch.n)
        next_lane_indices_np = np.array(batch.next_lane_index) # *** 获取下一个车道索引 ***

        # 转换为张量
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).to(self.device) # [batch_size]
        rewards = torch.FloatTensor(n_step_rewards_np).to(self.device) # [batch_size]
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.BoolTensor(dones_np).to(self.device) # [batch_size]
        is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)
        gammas = torch.FloatTensor(self.config.gamma ** n_np).to(self.device) # [batch_size]
        next_lane_indices = torch.LongTensor(next_lane_indices_np).to(self.device) # [batch_size]

        # --- 计算目标分布 (C51 逻辑) ---
        with torch.no_grad():
            # 从目标网络获取下一个状态分布
            if self.config.use_noisy_nets: self.target_net.reset_noise()
            next_dist_target = self.target_net(next_states) # [batch, action, atoms]

            # --- Double DQN 选择 ---
            if self.config.use_double_dqn:
                # 使用策略网络为下一个状态选择最佳动作
                if self.config.use_noisy_nets: self.policy_net.reset_noise()
                next_dist_policy = self.policy_net(next_states) # [batch, action, atoms]
                next_expected_q_policy = (next_dist_policy * self.support).sum(dim=2) # [batch, action]

                # *** 屏蔽 *下一个* 状态中无效动作的 Q 值 ***
                num_lanes = len(self.config.lane_max_speeds)
                q_values_masked = next_expected_q_policy.clone()
                # 如果在最左侧车道 (索引 0)，屏蔽左转
                q_values_masked[next_lane_indices == 0, 1] = -float('inf')
                # 如果在最右侧车道 (索引 num_lanes - 1)，屏蔽右转
                q_values_masked[next_lane_indices >= num_lanes - 1, 2] = -float('inf')

                # 根据策略网络 Q 值选择最佳有效动作
                best_next_actions = q_values_masked.argmax(dim=1) # [batch_size]

            else:
                 # 标准 DQN 选择 (使用目标网络)
                 next_expected_q_target = (next_dist_target * self.support).sum(dim=2) # [batch, action]
                 # 此处也应用屏蔽
                 num_lanes = len(self.config.lane_max_speeds)
                 q_values_masked = next_expected_q_target.clone()
                 q_values_masked[next_lane_indices == 0, 1] = -float('inf')
                 q_values_masked[next_lane_indices >= num_lanes - 1, 2] = -float('inf')
                 best_next_actions = q_values_masked.argmax(dim=1) # [batch_size]

            # 获取对应于所选最佳下一动作的目标网络分布
            best_next_dist = next_dist_target[range(self.config.batch_size), best_next_actions, :] # [batch, atoms]

            # --- 投影目标分布 ---
            # 计算投影支撑: Tz = R + gamma^N * z
            Tz = rewards.unsqueeze(1) + gammas.unsqueeze(1) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            # 计算投影的索引和偏移量
            b = (Tz - self.v_min) / self.delta_z
            lower_idx = b.floor().long()
            upper_idx = b.ceil().long()

            # 确保索引在边界处潜在的浮点问题后仍然有效
            lower_idx = lower_idx.clamp(min=0, max=self.num_atoms - 1)
            upper_idx = upper_idx.clamp(min=0, max=self.num_atoms - 1)

            # 使用 index_add_ 分布概率以提高效率
            target_dist_projected = torch.zeros_like(best_next_dist) # [batch, atoms]
            # 计算下部和上部索引的权重
            # 下部索引的权重 = (upper_idx - b)
            # 上部索引的权重 = (b - lower_idx)
            lower_weight = (upper_idx.float() - b).clamp(min=0, max=1) # 对下部 bin 的概率贡献
            upper_weight = (b - lower_idx.float()).clamp(min=0, max=1) # 对上部 bin 的概率贡献

            # 使用 index_add_ 进行投影 (比循环更高效)
            # 重塑以进行 index_add: 需要扁平的索引和扁平的值
            batch_indices = torch.arange(self.config.batch_size, device=self.device).unsqueeze(1).expand_as(lower_idx)

            # 沿维度 1 (原子) 扁平化张量以进行 index_add_
            flat_lower_idx = lower_idx.flatten()
            flat_upper_idx = upper_idx.flatten()
            flat_best_next_dist = best_next_dist.flatten()
            flat_lower_weight = lower_weight.flatten()
            flat_upper_weight = upper_weight.flatten()
            flat_batch_indices = batch_indices.flatten()

            # 投影到下部索引
            target_dist_projected.index_put_(
                (flat_batch_indices, flat_lower_idx),
                flat_best_next_dist * flat_lower_weight,
                accumulate=True
            )
            # 投影到上部索引
            target_dist_projected.index_put_(
                (flat_batch_indices, flat_upper_idx),
                flat_best_next_dist * flat_upper_weight,
                accumulate=True
            )


        # --- 计算所选动作的当前分布 ---
        if self.config.use_noisy_nets: self.policy_net.reset_noise()
        current_dist_all_actions = self.policy_net(states) # [batch, action, atoms]
        current_dist = current_dist_all_actions[range(self.config.batch_size), actions, :] # [batch, atoms]

        # --- 计算损失 (交叉熵) ---
        # 在 log 之前确保 current_dist 没有零
        elementwise_loss = -(target_dist_projected.detach() * (current_dist + 1e-7).log()).sum(dim=1) # [batch]

        # --- 计算 PER 的优先级 ---
        # 使用绝对 TD 误差 (逐元素损失) 作为优先级
        td_errors = elementwise_loss.detach()

        # 如果使用 PER，则更新缓冲区中的优先级
        if self.config.use_per and indices is not None:
             self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

        # --- 应用 IS 权重并计算最终损失 ---
        loss = (elementwise_loss * is_weights_tensor).mean()

        # --- 优化步骤 ---
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪有助于稳定 C51/NoisyNets
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.loss_history.append(loss.item())

        # --- 更新目标网络 ---
        if self.train_step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"---- 目标网络已更新 (步骤: {self.train_step_count}) ----") # 添加确认日志


        return loss.item()

#####################
#   主训练循环       #
#####################
def main():
    config = Config()

    # --- 基本检查和设置 ---
    if not os.path.exists(config.config_path): print(f"警告: 未找到 {config.config_path}。")

    # --- 环境和智能体初始化 ---
    env = SumoEnv(config)      # SUMO 环境 (内部处理归一化)
    agent = DQNAgent(config)   # DQN 智能体 (启用 C51/Noisy/PER/N-Step)

    # --- 日志记录和保存设置 ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = "dqn"
    if config.use_per: exp_name += "_per"
    if config.use_n_step: exp_name += f"_n{config.n_step}"
    if config.use_distributional: exp_name += "_c51"
    if config.use_noisy_nets: exp_name += "_noisy"
    exp_name += "_revised" # 指示修订版本
    results_dir = f"{exp_name}_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"结果将保存在: {results_dir}")
    print(f"配置: PER={config.use_per}, N-Step={config.n_step if config.use_n_step else 'Off'}, C51={config.use_distributional}, Noisy={config.use_noisy_nets}")
    print(f"LR={config.initial_learning_rate}, TargetUpdate={config.target_update_freq}, StartLearn={config.learning_starts}")
    print(f"奖励参数: FastLaneBonus={config.reward_faster_lane_bonus}, StaySlowPen={config.reward_staying_slow_penalty_scale}, LC_Penalty={config.reward_lane_change_penalty}, SpeedScale={config.reward_high_speed_scale}")


    # 保存此运行使用的配置
    try:
        config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        config_save_path = os.path.join(results_dir, "config_used.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
             json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
        print(f"配置已保存至: {config_save_path}")
    except Exception as e:
        print(f"警告: 无法保存配置 JSON: {e}")

    # 指标列表
    all_rewards_sum_norm = [] # 跟踪每回合归一化奖励的总和
    lane_change_counts = []
    collision_counts = []
    total_steps_per_episode = []
    avg_losses_per_episode = []
    # epsilon_values = [] # 不再与 Noisy Nets 一起使用
    beta_values = [] # 跟踪 PER beta
    best_avg_reward = -float('inf')
    global_step_count = 0

    # N-Step 缓冲区
    n_step_buffer = deque(maxlen=config.n_step)

    # --- DQN 训练循环 ---
    print("\n" + "#"*20 + f" 开始 DQN 训练 ({exp_name}) " + "#"*20)
    try:
        for episode in tqdm(range(1, config.dqn_episodes + 1), desc="DQN 训练回合"):
            state_norm = env.reset() # 获取初始归一化状态
            # 在开始回合前确保 state_norm 有效
            if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                print(f"错误: 回合 {episode} 的初始状态无效。再次重置...")
                time.sleep(1)
                state_norm = env.reset()
                if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                    print("致命错误: 重置后状态仍然无效。中止。")
                    raise RuntimeError("无效的初始状态。")

            episode_reward_norm_sum = 0.0 # 此回合归一化奖励的总和
            episode_loss_sum = 0.0
            episode_loss_count = 0
            done = False
            step_count = 0
            n_step_buffer.clear() # 为新回合清除 n-step 缓冲区

            while not done and step_count < config.max_steps:
                # 从 *原始* 状态获取当前车道索引以进行准确屏蔽
                current_lane_idx = int(round(env.last_raw_state[1]))
                num_lanes = len(config.lane_max_speeds)
                # 以防原始状态暂时无效，裁剪索引
                current_lane_idx = np.clip(current_lane_idx, 0, num_lanes - 1)

                # 根据归一化状态和当前车道获取动作
                action = agent.get_action(state_norm, current_lane_idx)

                # 步骤环境 -> 返回归一化的下一个状态、归一化的奖励、完成标志、下一个车道索引
                next_state_norm, norm_reward, done, next_lane_idx = env.step(action)

                 # 在添加到缓冲区之前验证来自 env.step 的数据
                if np.any(np.isnan(state_norm)) or np.any(np.isnan(next_state_norm)) or not np.isfinite(norm_reward):
                     print(f"警告: 在回合 {episode}, 步骤 {step_count} 检测到来自 env.step 的无效数据。跳过缓冲区推送。")
                     # 决定如何处理: 跳过步骤、结束回合、使用最后有效状态？
                     # 让我们跳过推送此转换，并继续使用最后有效状态
                     # 注意: 如果 step 严重失败，state_norm 在这里也可能无效
                     if np.any(np.isnan(next_state_norm)): next_state_norm = state_norm # 如果下一个状态错误，则保持旧状态
                     done = True # 如果状态变得无效，则结束回合
                     # 需要决定是否仍在此处处理 N-step 缓冲区...
                else:
                    # 将转换存储在 N-step 缓冲区中 (使用归一化的状态/奖励和下一个车道索引)
                    n_step_buffer.append(NStepTransition(state_norm, action, norm_reward, next_state_norm, done, next_lane_idx))

                    # --- N-Step Experience 生成 ---
                    # 如果缓冲区已满 或 回合结束 (并且缓冲区中有项目)，则生成 experience
                    if len(n_step_buffer) >= config.n_step or (done and len(n_step_buffer) > 0):
                        n_step_actual = len(n_step_buffer) # 此转换中的实际步数
                        n_step_return_discounted = 0.0
                        # 使用存储在缓冲区中的归一化奖励计算 N-step 回报
                        for i in range(n_step_actual):
                            # gamma^i * r_{t+i} (其中 r 是归一化奖励)
                            n_step_return_discounted += (config.gamma**i) * n_step_buffer[i].reward

                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        # 最终的下一个状态是缓冲区中最后一个转换的 s_{t+n}
                        s_t_plus_n = n_step_buffer[-1].next_state
                        # 如果缓冲区中的最后一个转换结束了回合，则达到终止状态
                        done_n_step = n_step_buffer[-1].done
                        # 获取对应于 s_t_plus_n 的车道索引
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index

                        exp = Experience(state=s_t, action=a_t, n_step_reward=n_step_return_discounted,
                                         next_state=s_t_plus_n, done=done_n_step, n=n_step_actual,
                                         next_lane_index=next_lane_idx_n_step) # 添加 next_lane_idx

                        # 将 experience 添加到回放缓冲区
                        try:
                           if config.use_per: agent.replay_buffer.push(exp)
                           else: agent.replay_buffer.append(exp)
                        except Exception as e_push:
                             print(f"将 experience 推送到缓冲区时出错: {e_push}")
                             traceback.print_exc()
                             # 决定是继续还是停止

                        # 如果因为缓冲区已满 (而不是 'done') 而生成了 N-step experience，则移除最旧的项目
                        if len(n_step_buffer) >= config.n_step:
                            n_step_buffer.popleft()

                        # 如果回合结束 ('done' 为 True)，缓冲区将在 while 循环之外稍后清除或在下面刷新。


                # 仅当步骤有效时才更新状态
                if not (np.any(np.isnan(next_state_norm)) or np.any(np.isinf(next_state_norm))):
                    state_norm = next_state_norm
                else:
                    # 处理无效的下一个状态 - 可能保持旧状态或终止
                    print(f"警告: 由于在回合 {episode}, 步骤 {step_count} 出现无效的 next_state_norm，保持先前状态")
                    # done = True # 可选地终止

                # 累积归一化奖励以进行日志记录
                episode_reward_norm_sum += norm_reward

                step_count += 1
                global_step_count += 1

                # 执行 DQN 更新
                loss = agent.update(global_step_count)
                if loss is not None:
                    episode_loss_sum += loss
                    episode_loss_count += 1

                # 显式检查碰撞标志 (如果设置了 done，则冗余，但安全)
                if env.collision_occurred: done = True

                # --- 如果回合结束，处理剩余的 N-step 转换 ---
                # 这需要改进: 如果 done，我们需要为缓冲区中所有剩余的项目生成 experiences
                if done and len(n_step_buffer) > 0:
                     # 持续生成 experiences 直到缓冲区为空
                     while len(n_step_buffer) > 0:
                        n_step_actual = len(n_step_buffer)
                        n_step_return_discounted = 0.0
                        for i in range(n_step_actual):
                            n_step_return_discounted += (config.gamma**i) * n_step_buffer[i].reward
                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        s_t_plus_n = n_step_buffer[-1].next_state
                        done_n_step = True # 回合结束
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index

                        exp = Experience(state=s_t, action=a_t, n_step_reward=n_step_return_discounted,
                                         next_state=s_t_plus_n, done=done_n_step, n=n_step_actual,
                                         next_lane_index=next_lane_idx_n_step)
                        try:
                           if config.use_per: agent.replay_buffer.push(exp)
                           else: agent.replay_buffer.append(exp)
                        except Exception as e_push:
                             print(f"将最终 experience 推送到缓冲区时出错: {e_push}")
                             traceback.print_exc()

                        n_step_buffer.popleft() # 移除已处理的转换


            # --- 回合结束 ---
            all_rewards_sum_norm.append(episode_reward_norm_sum) # 记录归一化奖励的总和
            lane_change_counts.append(env.change_lane_count)
            collision_counts.append(1 if env.collision_occurred else 0)
            total_steps_per_episode.append(step_count)
            avg_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0
            avg_losses_per_episode.append(avg_loss)

            # 获取 PER Beta 以进行日志记录
            current_beta = linear_decay(config.per_beta_start, config.per_beta_end,
                                        config.per_beta_annealing_steps, global_step_count)
            beta_values.append(current_beta)


            # --- 保存最佳模型 ---
            avg_window = 20 # 滚动平均窗口
            if episode >= avg_window:
                # 使用归一化奖励总和来跟踪最佳模型
                rewards_slice = all_rewards_sum_norm[max(0, episode - avg_window):episode]
                if rewards_slice:
                   current_avg_reward = np.mean(rewards_slice)
                   if current_avg_reward > best_avg_reward:
                       best_avg_reward = current_avg_reward
                       best_model_path = os.path.join(models_dir, "best_model.pth")
                       torch.save(agent.policy_net.state_dict(), best_model_path)
                       print(f"\n🎉 新的最佳平均奖励 ({avg_window}回合, 归一化总和): {best_avg_reward:.2f}! 模型已保存。")
                else: current_avg_reward = -float('inf')


            # --- 定期日志记录 ---
            if episode % config.log_interval == 0:
                 log_slice_start = max(0, episode - config.log_interval)
                 avg_reward_log = np.mean(all_rewards_sum_norm[log_slice_start:]) if all_rewards_sum_norm[log_slice_start:] else 0
                 avg_steps_log = np.mean(total_steps_per_episode[log_slice_start:]) if total_steps_per_episode[log_slice_start:] else 0
                 collision_rate_log = np.mean(collision_counts[log_slice_start:]) * 100 if collision_counts[log_slice_start:] else 0
                 avg_loss_log = np.mean(avg_losses_per_episode[log_slice_start:]) if avg_losses_per_episode[log_slice_start:] else 0
                 avg_lc_log = np.mean(lane_change_counts[log_slice_start:]) if lane_change_counts[log_slice_start:] else 0 # 记录换道次数

                 print(f"\n回合: {episode}/{config.dqn_episodes} | 平均奖励 (最后 {config.log_interval} 回合, 归一化总和): {avg_reward_log:.2f} "
                       f"| 最佳平均奖励 ({avg_window}回合): {best_avg_reward:.2f} "
                       f"| 平均步数: {avg_steps_log:.1f} | 平均换道: {avg_lc_log:.1f} " # 添加换道日志
                       f"| 碰撞率: {collision_rate_log:.1f}% "
                       f"| 平均损失: {avg_loss_log:.4f} | Beta: {current_beta:.3f} | 步数: {global_step_count}")

            # --- 定期模型保存 ---
            if episode % config.save_interval == 0:
                 periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                 torch.save(agent.policy_net.state_dict(), periodic_model_path)

    except KeyboardInterrupt: print("\n用户中断训练。")
    except Exception as e: print(f"\n训练期间发生致命错误: {e}"); traceback.print_exc()
    finally:
        print("正在关闭最终环境...")
        env._close()

        # --- 保存最终模型和数据 ---
        if 'agent' in locals() and agent is not None:
            last_model_path = os.path.join(models_dir, "last_model.pth")
            torch.save(agent.policy_net.state_dict(), last_model_path)
            print(f"最终模型已保存至: {last_model_path}")

            # --- 绘图 (根据要求修改) ---
            print("生成训练图表...")
            plt.figure("DQN 训练曲线 (修改版)", figsize=(12, 6)) # 修改图表名称和大小

            # 图 1: 奖励滚动平均 (来自原 Subplot 1)
            plt.subplot(1, 2, 1)
            plt.grid(True, linestyle='--')
            if len(all_rewards_sum_norm) >= 10:
                rolling_avg_reward = np.convolve(all_rewards_sum_norm, np.ones(10) / 10, mode='valid')
                plt.plot(np.arange(9, len(all_rewards_sum_norm)), rolling_avg_reward, label='10回合滚动平均奖励 (归一化总和)', color='red', linewidth=2)
            else:
                plt.plot([], label='10回合滚动平均奖励 (数据不足)', color='red', linewidth=2) # 占位符
            plt.title("滚动平均回合奖励")
            plt.xlabel("回合")
            plt.ylabel("平均总奖励 (归一化)")
            plt.legend()

            # 图 2: 损失滚动平均 (来自原 Subplot 4)
            plt.subplot(1, 2, 2)
            plt.grid(True, linestyle='--')
            valid_losses = [l for l in avg_losses_per_episode if l > 0]
            valid_indices = [i for i, l in enumerate(avg_losses_per_episode) if l > 0]
            if len(valid_losses) >= 10:
                 rolling_loss_avg = np.convolve(valid_losses, np.ones(10) / 10, mode='valid')
                 plt.plot(valid_indices[9:], rolling_loss_avg, label='10回合滚动平均损失 (交叉熵)', color='purple', linewidth=1.5)
            else:
                 plt.plot([], label='10回合滚动平均损失 (数据不足)', color='purple', linewidth=1.5) # 占位符
            plt.title("滚动平均 DQN 损失")
            plt.xlabel("回合")
            plt.ylabel("平均损失 (交叉熵)")
            plt.legend()
            # 可以考虑对数刻度: plt.yscale('log')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plot_path = os.path.join(results_dir, "dqn_training_curves_revised_simple.png") # 新的文件名
            plt.savefig(plot_path)
            plt.close("all")
            print(f"训练图表已保存至: {plot_path}")

            # --- 保存训练数据 JSON ---
            print("正在保存训练数据...")
            training_data = {
                "episode_rewards_sum_norm": all_rewards_sum_norm, # 名称反映了它是归一化奖励的总和
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "avg_losses_per_episode": avg_losses_per_episode,
                "beta_values_per_episode": beta_values,
                "detailed_loss_history_per_step": agent.loss_history
            }
            data_path = os.path.join(results_dir, "training_data_final_revised.json")
            try:
                # 修复: 定义健壮的序列化器，不使用 self
                def default_serializer(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, np.bool_): return bool(obj)
                    elif isinstance(obj, (datetime.datetime, datetime.date)): return obj.isoformat()
                    # 默认回退，处理无法直接序列化的类型
                    return str(obj)

                # 清理数据以确保存储为基本类型
                for key in training_data:
                     if isinstance(training_data[key], list) and len(training_data[key]) > 0:
                         # 转换列表中的 numpy 类型
                         training_data[key] = [default_serializer(item) for item in training_data[key]]
                         # 处理损失中潜在的 None 或 NaN
                         if key == "avg_losses_per_episode" or key == "detailed_loss_history_per_step":
                              training_data[key] = [0.0 if (item is None or not np.isfinite(item)) else float(item) for item in training_data[key]]


                with open(data_path, "w", encoding="utf-8") as f:
                     # 在 dump 调用中直接使用 default 参数
                     json.dump(training_data, f, indent=4, ensure_ascii=False, default=default_serializer)
                print(f"训练数据已保存至: {data_path}")
            except Exception as e:
                print(f"保存训练数据时出错: {e}")
                traceback.print_exc()
        else:
            print("智能体未初始化，无法保存最终模型/数据。")

        print(f"\n DQN 训练 ({exp_name}) 完成。结果已保存在: {results_dir}")

if __name__ == "__main__":
    if not os.path.exists("a.sumocfg"): print("警告: 未找到 a.sumocfg。")
    if not os.path.exists("a.net.xml"): print("警告: 未找到 a.net.xml。")
    if not os.path.exists("a.rou.xml"): print("警告: 未找到 a.rou.xml。")
    main()