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
import math # For checking adjacent lane safety

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

#####################
#     配置区域       #
#####################
class Config:
    # --- SUMO 配置 --- (保持不变)
    sumo_binary = "sumo" # 或 "sumo-gui"
    config_path = "a.sumocfg"
    step_length = 0.2
    ego_vehicle_id = "drl_ego_car"
    ego_type_id = "car_ego"
    port_range = (8890, 8900)

    # --- 行为克隆 (BC) --- (保持不变)
    use_bc = True
    bc_collect_episodes = 15
    bc_epochs = 20
    bc_learning_rate = 1e-4

    # --- PPO 训练 --- (保持不变)
    ppo_episodes = 500
    max_steps = 8000
    log_interval = 10
    save_interval = 50

    # --- PPO 超参数 --- (保持不变, LR decay is handled)
    gamma = 0.99
    clip_epsilon = 0.2
    initial_learning_rate = 3e-4
    final_learning_rate = 1e-6
    batch_size = 512
    ppo_epochs = 5
    hidden_size = 256 # Keep 256 for now, increase if needed after reward changes
    gae_lambda = 0.95
    value_clip = True
    value_clip_epsilon = 0.2
    normalize_advantages = True
    gradient_clip_norm = 1.0

    # --- 归一化 --- (保持不变)
    normalize_observations = True
    normalize_rewards = True
    obs_norm_clip = 5.0
    reward_norm_clip = 10.0
    norm_update_rate = 0.001

    # --- 熵正则化 --- (保持不变)
    use_entropy_decay = True
    entropy_coef_start = 0.05
    entropy_coef_end = 0.005
    entropy_decay_episodes = int(ppo_episodes * 0.8)

    # --- 状态/动作空间 --- (保持不变)
    state_dim = 12
    action_dim = 3

    # --- 环境参数 --- (保持不变)
    max_speed_global = 33.33
    max_distance = 100.0
    lane_max_speeds = [33.33, 27.78, 22.22] # Corresponds to E0_0, E0_1, E0_2 in a.net.xml
    num_train_lanes = len(lane_max_speeds)

    ### <<< 优化开始 >>> ###
    # --- 奖励函数参数 (更强的换道激励) ---
    reward_collision = -100.0                # 保持高碰撞惩罚
    reward_high_speed_scale = 0.20           # 适度奖励相对于车道限制的高速 (原为 0.15?)
    reward_low_speed_penalty_scale = 0.1     # 对过慢的小惩罚
    reward_inefficient_change_penalty = -0.5 # *** 增加的惩罚 *** - 移动到 *较慢* 潜在车道
    reward_faster_lane_bonus = 1.2           # *** 大幅增加的奖励 *** - 移动到 *较快* 潜在车道 (原为 0.1 或 0.05?)
    reward_staying_slow_penalty_scale = 0.4  # *** 增加的惩罚 *** - *未* 换到可用更快车道 (原为 0.1?)
    reward_fast_lane_preference = 0.08       # *** 新增: 持续奖励处于更快车道 (较低索引) ***
    time_alive_reward = 0.005                # *** 略微减少以使其他奖励更具影响力 *** (原为 0.01?)
    reward_comfort_penalty_scale = 0.05      # 保持舒适度惩罚 (急刹车)
    target_speed_factor = 0.95               # 相对于车道最大速度的目标速度
    safe_distance_penalty_scale = 0.2        # 保持与前车过近的惩罚
    min_buffer_dist_reward = 5.0             # 基本安全距离
    time_gap_reward = 0.8                    # 计算动态安全距离的时间间隔

    # 奖励函数中考虑换道机会的安全距离
    # 用于计算 'staying_slow_penalty'
    min_safe_change_dist_front = 20.0        # 目标车道前方车辆所需的间隙
    min_safe_change_dist_back = 10.0         # 目标车道后方车辆所需的间隙
    ### <<< 优化结束 >>> ###

    # BC 安全参数 (保持不变)
    min_buffer_dist_bc = 6.0
    time_gap_bc = 1.8


#####################
#   归一化          # (保持不变)
#####################
class RunningMeanStd:
    """计算运行均值和标准差"""
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4, alpha: float = 0.001):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.alpha = alpha

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1.0 - self.alpha) * self.var + self.alpha * batch_var
        self.count += batch_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

class RewardNormalizer:
    def __init__(self, gamma: float, epsilon: float = 1e-8, alpha: float = 0.001):
        self.returns = collections.deque(maxlen=1000) # 存储最近的回报以计算方差
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.epsilon = epsilon
        self.alpha = alpha # 用于更新均值/方差的 EMA alpha

    def update(self, rewards: np.ndarray):
        """使用 EMA 更新回报的均值和方差估计。"""
        self.returns.extend(rewards) # 添加新的回合/批量回报
        if len(self.returns) > 1:
             current_mean = np.mean(self.returns)
             current_var = np.var(self.returns)
             # EMA 更新
             self.ret_mean = (1.0 - self.alpha) * self.ret_mean + self.alpha * current_mean
             self.ret_var = (1.0 - self.alpha) * self.ret_var + self.alpha * current_var

    def normalize(self, r: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """使用运行方差归一化奖励。"""
        std = np.sqrt(self.ret_var + self.epsilon)
        norm_r = r / std
        return np.clip(norm_r, -clip, clip)


#####################
#   工具函数         # (保持不变)
#####################
def get_available_port(start_port, end_port):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError: continue
    raise IOError(f"在范围 [{start_port}, {end_port}] 内未找到可用端口。")

def kill_sumo_processes():
    killed = False
    try:
        if os.name == 'nt':
            result1 = os.system("taskkill /f /im sumo.exe >nul 2>&1")
            result2 = os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
            killed = (result1 == 0 or result2 == 0)
        else:
            result1 = os.system("pkill -f sumo > /dev/null 2>&1")
            result2 = os.system("pkill -f sumo-gui > /dev/null 2>&1")
            killed = (result1 == 0 or result2 == 0)
    except Exception as e: print(f"终止 SUMO 进程时出错: {e}")
    time.sleep(0.1)

def linear_decay(start_val, end_val, total_steps, current_step):
    if current_step >= total_steps: return end_val
    return start_val + (end_val - start_val) * (current_step / total_steps)


#####################
#   SUMO 环境封装    #
#####################
class SumoEnv:
    # __init__, reset_metrics, _start_sumo, _add_ego_vehicle, reset,
    # _get_surrounding_vehicle_info, _get_state, _close
    # 保持不变。 step() 调用新的 _calculate_reward_optimized。

    def __init__(self, config: Config):
        self.config = config
        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_speed = 0.0
        self.last_raw_state = np.zeros(config.state_dim)
        self.last_lane_idx = 0 # 存储 *上一步* 的车道索引
        self.reset_metrics()

    def reset_metrics(self):
        self.change_lane_count = 0
        self.collision_occurred = False
        self.current_step = 0
        self.last_action = 0 # 保持车道 = 0

    def _start_sumo(self):
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.config.port_range[0], self.config.port_range[1])
        except IOError as e:
             print(f"错误: 无法找到可用端口: {e}"); sys.exit(1)

        sumo_cmd = [
            self.config.sumo_binary, "-c", self.config.config_path,
            "--remote-port", str(self.traci_port), "--step-length", str(self.config.step_length),
            "--collision.check-junctions", "true", "--collision.action", "warn", # 检测碰撞
            "--time-to-teleport", "-1", "--no-warnings", "true",
            "--seed", str(np.random.randint(0, 10000))
        ]
        try:
             stdout_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
        except FileNotFoundError: print(f"错误: SUMO 可执行文件 '{self.config.sumo_binary}' 未找到。"); sys.exit(1)
        except Exception as e: print(f"错误: 无法启动 SUMO 进程: {e}"); sys.exit(1)

        for attempt in range(5):
            try:
                time.sleep(1.0 + attempt * 0.5)
                traci.init(self.traci_port)
                print(f"✅ SUMO TraCI 已连接 (端口: {self.traci_port}).")
                return
            except traci.exceptions.TraCIException:
                if attempt == 4: print("达到最大 TraCI 连接尝试次数。"); self._close(); raise ConnectionError(f"无法连接到 SUMO (端口: {self.traci_port})。")
            except Exception as e: print(f"连接 TraCI 时发生意外错误: {e}"); self._close(); raise ConnectionError(f"连接到 SUMO 时发生未知错误 (端口: {self.traci_port})。")

    def _add_ego_vehicle(self):
        ego_route_id = "route_E0"
        if ego_route_id not in traci.route.getIDList():
             try: traci.route.add(ego_route_id, ["E0"])
             except traci.exceptions.TraCIException as e: raise RuntimeError(f"添加路径 '{ego_route_id}' 失败: {e}")

        if self.config.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.config.ego_type_id)
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0")
                # 此处保留原始 PPO BC 参数
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcSpeedGain", "1.5") # 更愿意为速度而改变
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcCooperative", "0.5") # 适度合作
            except traci.exceptions.TraCIException as e: print(f"警告: 设置 Ego 类型 '{self.config.ego_type_id}' 参数失败: {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            try: traci.vehicle.remove(self.config.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"警告: 移除残留 Ego 失败: {e}")

        try:
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id, typeID=self.config.ego_type_id, depart="now", departLane="random", departSpeed="max")
            wait_steps = int(2.0 / self.config.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.config.ego_vehicle_id in traci.vehicle.getIDList(): ego_appeared = True; break
            if not ego_appeared: raise RuntimeError(f"Ego 车辆在 {wait_steps} 步内未出现。")
        except traci.exceptions.TraCIException as e: raise RuntimeError(f"添加 Ego 车辆失败: {e}")


    def reset(self) -> np.ndarray:
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.reset_metrics()
        self.last_speed = 0.0
        self.last_raw_state = np.zeros(self.config.state_dim)
        initial_state = np.zeros(self.config.state_dim)
        self.last_lane_idx = 0 # 初始化车道索引

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
             try:
                 self.last_speed = traci.vehicle.getSpeed(self.config.ego_vehicle_id)
                 initial_state = self._get_state() # 获取原始状态
                 self.last_raw_state = initial_state.copy()
                 self.last_lane_idx = int(round(initial_state[1])) # 设置初始车道索引
             except traci.exceptions.TraCIException: print("警告: 在 reset 中的初始状态获取期间发生 TraCI 异常。")
             except IndexError: print("警告: 访问初始状态时发生 IndexError - 可能是无效的状态数组。")
             except Exception as e_reset: print(f"警告: 获取初始状态时发生未知错误: {e_reset}")
        else: print("警告: 在 reset 中的 add 后未立即找到 Ego。")

        return self.last_raw_state # 返回原始状态

    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        max_dist = self.config.max_distance
        infos = {'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0), 'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0), 'right_back': (max_dist, 0.0)}
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        try: # 在此处添加 try-except 块以提高鲁棒性
            ego_road = traci.vehicle.getRoadID(ego_id)
            if not ego_road or not ego_road.startswith("E"): return infos # 检查是否在有效的路段上

            num_lanes_on_edge = self.config.num_train_lanes # 使用配置值

            for veh_id in veh_ids:
                if veh_id == ego_id: continue
                try:
                    if traci.vehicle.getRoadID(veh_id) != ego_road: continue
                    veh_pos = traci.vehicle.getPosition(veh_id)
                    veh_lane = traci.vehicle.getLaneIndex(veh_id)
                    veh_speed = traci.vehicle.getSpeed(veh_id)

                    # 在比较之前确保 veh_lane 有效
                    if not (0 <= veh_lane < num_lanes_on_edge): continue

                    dx = veh_pos[0] - ego_pos[0]; distance = abs(dx) # 纵向距离
                    if distance >= max_dist: continue

                    rel_speed = ego_speed - veh_speed # 相对速度 (Ego - Other)

                    if veh_lane == ego_lane: # 同一车道
                        if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                    elif veh_lane == ego_lane - 1: # 左车道
                        if dx > 0 and distance < infos['left_front'][0]: infos['left_front'] = (distance, rel_speed)
                        elif dx <= 0 and distance < infos['left_back'][0]: infos['left_back'] = (distance, rel_speed) # 对后方使用简单的 dx <= 0
                    elif veh_lane == ego_lane + 1: # 右车道
                        if dx > 0 and distance < infos['right_front'][0]: infos['right_front'] = (distance, rel_speed)
                        elif dx <= 0 and distance < infos['right_back'][0]: infos['right_back'] = (distance, rel_speed) # 对后方使用简单的 dx <= 0
                except traci.exceptions.TraCIException: continue # 跳过有问题的车辆
        except traci.exceptions.TraCIException as e_outer:
            # 处理 ego 车辆本身导致问题的情况 (例如，在检查中途消失)
            if "Vehicle '" + ego_id + "' is not known" not in str(e_outer):
                print(f"警告: 获取 {ego_id} 周围信息时发生 TraCI 错误: {e_outer}")
            return infos # 返回默认信息

        return infos


    def _get_state(self) -> np.ndarray:
        """获取原始状态向量"""
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id
        if ego_id not in traci.vehicle.getIDList(): return self.last_raw_state
        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road or not current_road.startswith("E"): return self.last_raw_state # 不在预期的路段上

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            num_lanes = self.config.num_train_lanes

            # 验证车道索引
            if not (0 <= ego_lane < num_lanes):
                 print(f"警告: 在 _get_state 中检测到无效的 ego 车道 {ego_lane}。进行裁剪。")
                 ego_lane = np.clip(ego_lane, 0, num_lanes - 1)

            # 使用 TraCI 的 couldChangeLane 进行安全检查
            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # 填充状态向量
            state[0] = ego_speed
            state[1] = float(ego_lane)
            state[2] = min(surround_info['front'][0], self.config.max_distance)
            state[3] = surround_info['front'][1] # 相对速度可以为负
            state[4] = min(surround_info['left_front'][0], self.config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.config.max_distance) # 后方仅距离
            state[7] = min(surround_info['right_front'][0], self.config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.config.max_distance) # 后方仅距离
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                 print("警告: 在状态计算中检测到 NaN 或 Inf。使用最后有效状态。")
                 return self.last_raw_state

            self.last_raw_state = state.copy() # 更新最后已知的良好原始状态

        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" not in str(e):
                print(f"警告: 获取 {ego_id} 状态时发生 TraCI 错误: {e}。")
            return self.last_raw_state # 出错时返回最后状态
        except Exception as e:
            print(f"警告: 获取 {ego_id} 状态时发生未知错误: {e}。"); traceback.print_exc()
            return self.last_raw_state

        return state # 返回当前原始状态

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行一步，返回 (next_raw_state, raw_reward, done)"""
        done = False; raw_reward = 0.0
        ego_id = self.config.ego_vehicle_id
        self.last_action = action # 存储动作

        if ego_id not in traci.vehicle.getIDList():
             self.collision_occurred = True # 如果在步骤开始时缺失，则假设发生碰撞
             return self.last_raw_state, self.config.reward_collision, True

        try:
            # 稍后为奖励计算存储动作/步骤 *之前* 的车道索引
            previous_lane_idx = self.last_lane_idx

            current_lane = traci.vehicle.getLaneIndex(ego_id)
            num_lanes = self.config.num_train_lanes
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1) # 如果无效则裁剪

            # --- 执行动作 ---
            lane_change_requested = False
            if action == 1 and current_lane > 0: # 左转
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
                lane_change_requested = True
            elif action == 2 and current_lane < (num_lanes - 1): # 右转
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
                lane_change_requested = True
            # action 0: 保持车道 - 什么都不做

            # --- 仿真步骤 ---
            traci.simulationStep()
            self.current_step += 1

            # --- 步骤之后检查状态 ---
            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True; done = True
                raw_reward = self.config.reward_collision
                next_state_raw = self.last_raw_state # 如果 ego 消失，则返回最后状态
                # 如果可能，根据最后已知状态更新 last_lane_idx
                self.last_lane_idx = int(round(self.last_raw_state[1])) if len(self.last_raw_state) > 1 else 0
                return next_state_raw, raw_reward, done

            # 检查 SUMO 碰撞警告
            collisions = traci.simulation.getCollisions()
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True; done = True
                    raw_reward = self.config.reward_collision; break

            # --- 获取下一个状态 (原始) ---
            next_state_raw = self._get_state() # 获取步骤 *之后* 的当前状态
            current_lane_after_step = int(round(next_state_raw[1])) # 根据新状态更新当前车道


            # --- 计算奖励 (除非发生碰撞) ---
            if not self.collision_occurred:
                current_speed_after_step = next_state_raw[0]
                front_dist_after_step = next_state_raw[2]
                lf_dist_after_step = next_state_raw[4]; lb_dist_after_step = next_state_raw[6]
                rf_dist_after_step = next_state_raw[7]; rb_dist_after_step = next_state_raw[9]
                can_change_left_after_step = next_state_raw[10] > 0.5
                can_change_right_after_step = next_state_raw[11] > 0.5

                actual_lane_change = (previous_lane_idx != current_lane_after_step)
                if actual_lane_change: self.change_lane_count += 1

                # *** 调用更新后的奖励函数 ***
                raw_reward = self._calculate_reward_optimized(
                    action_taken=action, # 此步骤尝试的动作
                    current_speed=current_speed_after_step,
                    current_lane=current_lane_after_step,
                    front_dist=front_dist_after_step,
                    previous_lane=previous_lane_idx, # 步骤 *之前* 的车道索引
                    can_change_left=can_change_left_after_step,
                    can_change_right=can_change_right_after_step,
                    left_front_dist=lf_dist_after_step, left_back_dist=lb_dist_after_step,
                    right_front_dist=rf_dist_after_step, right_back_dist=rb_dist_after_step,
                    lane_change_requested=lane_change_requested # 我们是否要求 SUMO 换道？
                )

            # 检查其他终止条件
            if traci.simulation.getTime() >= 3600: done = True # 最大仿真时间
            if self.current_step >= self.config.max_steps: done = True # 最大回合步数

        except traci.exceptions.TraCIException as e:
            print(f"错误: 在步骤 {self.current_step} 期间发生 TraCI 异常: {e}")
            if ego_id not in traci.vehicle.getIDList(): self.collision_occurred = True
            raw_reward = self.config.reward_collision; done = True
            next_state_raw = self.last_raw_state
        except Exception as e:
            print(f"错误: 在步骤 {self.current_step} 期间发生未知异常: {e}"); traceback.print_exc()
            done = True; raw_reward = self.config.reward_collision
            self.collision_occurred = True; next_state_raw = self.last_raw_state

        # 更新 *下一个* 步骤计算的最后已知值
        self.last_speed = next_state_raw[0]
        self.last_lane_idx = int(round(next_state_raw[1])) # 根据步骤 *之后* 的状态更新车道索引

        return next_state_raw, raw_reward, done


    ### <<< 优化开始 >>> ###
    # --- 优化的奖励计算 ---
    def _calculate_reward_optimized(self, action_taken: int, current_speed: float, current_lane: int,
                                    front_dist: float, previous_lane: int,
                                    can_change_left: bool, can_change_right: bool,
                                    left_front_dist: float, left_back_dist: float,
                                    right_front_dist: float, right_back_dist: float,
                                    lane_change_requested: bool) -> float:
        """
        计算奖励，对有益的换道给予更强的激励。
        使用仿真步骤之后的状态信息。
        """
        # 碰撞奖励在调用此函数之前的 step 函数中处理
        if self.collision_occurred:
             # 如果 collision=True，则不应调用此函数，但作为安全措施
             return 0.0

        try:
            num_lanes = self.config.num_train_lanes
            # 以防车道索引无效，对其进行裁剪
            current_lane = np.clip(current_lane, 0, num_lanes - 1)
            previous_lane = np.clip(previous_lane, 0, num_lanes - 1)

            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            # 1. 速度奖励 / 惩罚
            speed_diff = abs(current_speed - target_speed)
            # 奖励接近目标速度 (指数衰减)
            speed_reward = np.exp(- (speed_diff / (target_speed * 0.4 + 1e-6))**2 ) * self.config.reward_high_speed_scale
            # 惩罚显著低于目标速度
            low_speed_penalty = 0.0
            if current_speed < target_speed * 0.6:
                # 线性惩罚从 0 缩放到 -reward_low_speed_penalty_scale
                low_speed_penalty = (current_speed / (target_speed * 0.6) - 1.0) * self.config.reward_low_speed_penalty_scale

            # 2. 安全距离惩罚 (前车)
            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < self.config.max_distance and front_dist < min_safe_dist:
                 # 线性惩罚从 0 缩放到 -reward_safe_distance_penalty_scale
                 safety_dist_penalty = max(-1.0, (front_dist / min_safe_dist - 1.0)) * self.config.safe_distance_penalty_scale

            # 3. 舒适度惩罚 (急刹车)
            comfort_penalty = 0.0
            acceleration = (current_speed - self.last_speed) / self.config.step_length
            # 惩罚比 -3 m/s^2 更严重的减速
            if acceleration < -3.0:
                # 对更严厉的刹车进行线性惩罚缩放
                comfort_penalty = (acceleration + 3.0) * self.config.reward_comfort_penalty_scale # 负值

            # 4. 时间存活奖励 (生存的小正奖励)
            time_alive = self.config.time_alive_reward

            # --- 增强的换道激励 ---
            lane_change_bonus = 0.0             # 移动到更快车道的奖励
            staying_slow_penalty = 0.0          # 未移动到可用更快车道的惩罚
            inefficient_change_penalty = 0.0    # 移动到较慢车道的惩罚
            fast_lane_preference_reward = 0.0   # 持续奖励处于更快车道

            # 检查上一步和这一步之间是否实际发生了换道
            actual_lane_change = (current_lane != previous_lane)

            # 基于实际换道完成的奖励/惩罚
            if actual_lane_change:
                previous_max_speed = self.config.lane_max_speeds[previous_lane]
                current_max_speed = self.config.lane_max_speeds[current_lane] # *新* 车道的最大速度
                # 如果新车道有更高的速度限制，则奖励
                if current_max_speed > previous_max_speed:
                    lane_change_bonus = self.config.reward_faster_lane_bonus # 增加奖励
                # 如果新车道有更低的速度限制，则惩罚
                elif current_max_speed < previous_max_speed:
                    inefficient_change_penalty = self.config.reward_inefficient_change_penalty # 增加惩罚

            # 如果未发生换道，则因保持慢速而惩罚
            else: # 此步骤未发生换道 (current_lane == previous_lane)
                # 检查从向左换道获得的潜在收益
                left_lane_idx = current_lane - 1
                if left_lane_idx >= 0 and self.config.lane_max_speeds[left_lane_idx] > lane_max_speed:
                    # 左车道更快，检查根据奖励阈值换道是否安全
                    is_left_safe_for_reward = (can_change_left and # SUMO 认为可能
                                    left_front_dist > self.config.min_safe_change_dist_front and
                                    left_back_dist > self.config.min_safe_change_dist_back)
                    if is_left_safe_for_reward:
                        # 对未选择更快、安全的左侧选项应用惩罚
                        staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale # 增加惩罚

                # 检查从向右换道获得的潜在收益 (仅当未因左侧而被惩罚时)
                right_lane_idx = current_lane + 1
                if staying_slow_penalty == 0.0 and right_lane_idx < num_lanes and self.config.lane_max_speeds[right_lane_idx] > lane_max_speed:
                    # 右车道更快，检查换道是否安全
                    is_right_safe_for_reward = (can_change_right and # SUMO 认为可能
                                     right_front_dist > self.config.min_safe_change_dist_front and
                                     right_back_dist > self.config.min_safe_change_dist_back)
                    if is_right_safe_for_reward:
                        # 对未选择更快、安全的右侧选项应用惩罚
                        staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale # 增加惩罚

            # *** 新增: 快车道偏好奖励 ***
            # 为处于具有更高速度限制 (较低索引) 的车道提供持续奖励
            if num_lanes > 1:
                # 线性缩放: 最快车道 (索引 0) 为 1，最慢车道 (索引 N-1) 为 0
                preference_factor = (num_lanes - 1 - current_lane) / (num_lanes - 1)
                fast_lane_preference_reward = self.config.reward_fast_lane_preference * preference_factor

            # --- 总奖励 ---
            total_reward = (speed_reward + low_speed_penalty +
                            safety_dist_penalty + comfort_penalty +
                            time_alive +
                            lane_change_bonus + inefficient_change_penalty + staying_slow_penalty +
                            fast_lane_preference_reward) # 添加新的偏好奖励

            return total_reward

        except IndexError as e_idx:
            print(f"警告: 计算奖励时发生 IndexError (可能是无效的车道索引 {current_lane} 或 {previous_lane}): {e_idx}。返回 0。")
            traceback.print_exc()
            return 0.0
        except Exception as e:
            print(f"警告: 计算奖励时出错: {e}。返回 0。"); traceback.print_exc()
            return 0.0
    ### <<< 优化结束 >>> ###

    def _close(self):
        if self.sumo_process:
            try: traci.close()
            except Exception: pass
            finally:
                try:
                    if self.sumo_process.poll() is None: self.sumo_process.terminate(); self.sumo_process.wait(timeout=2)
                except subprocess.TimeoutExpired: self.sumo_process.kill(); self.sumo_process.wait(timeout=1)
                except Exception as e: print(f"警告: SUMO 终止期间出错: {e}")
                self.sumo_process = None; self.traci_port = None; time.sleep(0.1)
        else: self.traci_port = None

# --- 剩余代码 (rule_based_action_improved, BehaviorCloningNet, bc_train, PPO, Agent, main) ---
# --- 与您原始提供的 ppoplus.py 完全相同 ---
# --- 将 ppoplus.py 的其余代码复制到此行下方 ---

#####################
#   规则策略 (BC用)  # (保持不变)
#####################
def rule_based_action_improved(state: np.ndarray, config: Config) -> int:
    EGO_SPEED, LANE_IDX, FRONT_DIST, FRONT_REL_SPEED, \
    LF_DIST, LF_REL_SPEED, LB_DIST, \
    RF_DIST, RF_REL_SPEED, RB_DIST, \
    CAN_LEFT, CAN_RIGHT = range(config.state_dim)
    ego_speed = state[EGO_SPEED]; lane_idx = int(round(state[LANE_IDX]))
    front_dist = state[FRONT_DIST]; front_rel_speed = state[FRONT_REL_SPEED]
    lf_dist = state[LF_DIST]; lb_dist = state[LB_DIST]
    rf_dist = state[RF_DIST]; rb_dist = state[RB_DIST]
    can_change_left = state[CAN_LEFT] > 0.5; can_change_right = state[CAN_RIGHT] > 0.5
    num_lanes = config.num_train_lanes
    if not (0 <= lane_idx < num_lanes): lane_idx = np.clip(lane_idx, 0, num_lanes - 1)
    current_max_speed = config.lane_max_speeds[lane_idx]
    left_max_speed = config.lane_max_speeds[lane_idx - 1] if lane_idx > 0 else -1
    right_max_speed = config.lane_max_speeds[lane_idx + 1] if lane_idx < (num_lanes - 1) else -1
    reaction_time_gap = config.time_gap_bc; min_buffer_dist = config.min_buffer_dist_bc
    safe_follow_dist = min_buffer_dist + ego_speed * reaction_time_gap
    required_front_gap = min_buffer_dist + ego_speed * (reaction_time_gap * 0.9)
    required_back_gap = min_buffer_dist + ego_speed * (reaction_time_gap * 0.6)
    emergency_dist = min_buffer_dist + ego_speed * (reaction_time_gap * 0.4)
    # 紧急操作 (前方即将发生碰撞)
    if front_dist < emergency_dist and front_rel_speed > 2.0: # 前车显著较慢
        left_safe_emergency = can_change_left and (lf_dist > required_front_gap * 0.7) and (lb_dist > required_back_gap * 0.7)
        right_safe_emergency = can_change_right and (rf_dist > required_front_gap * 0.7) and (rb_dist > required_back_gap * 0.7)
        if left_safe_emergency: return 1 # 紧急向左换道
        if right_safe_emergency: return 2 # 紧急向右换道
        return 0 # 没有安全选项，保持车道 (可能急刹车)
    # 跟车过近
    if front_dist < safe_follow_dist and front_rel_speed > 1.0: # 前车较慢
        left_safe = can_change_left and (lf_dist > required_front_gap) and (lb_dist > required_back_gap)
        right_safe = can_change_right and (rf_dist > required_front_gap) and (rb_dist > required_back_gap)
        # 如果安全，优先换到潜在更快的车道
        prefer_left = left_safe and (left_max_speed > right_max_speed or not right_safe)
        prefer_right = right_safe and (right_max_speed > left_max_speed or not left_safe)
        if prefer_left: return 1
        if prefer_right: return 2
        # 如果速度相似或只有一个选项安全，则选择安全的那个
        if left_safe: return 1
        if right_safe: return 2
        return 0 # 没有安全换道，保持车道
    # 移动到更快车道的机会 (速度增益)
    speed_threshold = current_max_speed * 0.85 # 如果低于最大车道速度的 85%，则考虑换道
    if ego_speed < speed_threshold:
        # 如果左车道显著更快，则检查左车道
        if can_change_left and left_max_speed > current_max_speed * 1.05: # 左车道快 5% 以上
            left_extra_safe = (lf_dist > required_front_gap * 1.5) and (lb_dist > required_back_gap * 1.2) # 需要更大的间隙
            if left_extra_safe: return 1
        # 如果右车道显著更快，则检查右车道
        if can_change_right and right_max_speed > current_max_speed * 1.05: # 右车道快 5% 以上
            right_extra_safe = (rf_dist > required_front_gap * 1.5) and (rb_dist > required_back_gap * 1.2) # 需要更大的间隙
            if right_extra_safe: return 2
    # 默认: 保持车道
    return 0

#####################
#   BC Actor 网络    # (保持不变)
#####################
class BehaviorCloningNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(BehaviorCloningNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.actor(x)

def bc_train(config: Config, bc_data: List[Tuple[np.ndarray, int]], obs_normalizer: Optional[RunningMeanStd]) -> Optional[BehaviorCloningNet]:
    if not config.use_bc or not bc_data: print("跳过 BC 训练。"); return None
    print(f"\n--- 开始 BC 训练 ({len(bc_data)} 个样本) ---")
    net = BehaviorCloningNet(config.state_dim, config.action_dim, config.hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=config.bc_learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    try:
        states_raw = np.array([d[0] for d in bc_data])
        actions = torch.LongTensor(np.array([d[1] for d in bc_data]))
        # 如果启用，使用提供的归一化器应用归一化
        if config.normalize_observations and obs_normalizer:
             # 使用 BC 数据更新归一化器
             obs_normalizer.update(states_raw)
             # 归一化状态以进行训练
             states_normalized = (states_raw - obs_normalizer.mean) / (obs_normalizer.std + 1e-8)
             states_normalized = np.clip(states_normalized, -config.obs_norm_clip, config.obs_norm_clip)
             states_tensor = torch.FloatTensor(states_normalized)
             print("BC 数据已使用运行统计信息进行归一化。")
        else: states_tensor = torch.FloatTensor(states_raw) # 如果归一化关闭，则使用原始状态
    except Exception as e: print(f"准备 BC 数据时出错: {e}"); return None
    dataset = torch.utils.data.TensorDataset(states_tensor, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    net.train()
    all_bc_losses = []
    for epoch in range(config.bc_epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            logits = net(batch_states); loss = loss_fn(logits, batch_actions)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader); all_bc_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == config.bc_epochs - 1: print(f"[BC] Epoch {epoch+1}/{config.bc_epochs}, 平均损失 = {avg_loss:.6f}")
    net.eval(); print("BC 训练完成。")
    # 保存损失曲线图
    plt.figure("BC Loss Curve", figsize=(8, 4)); plt.plot(all_bc_losses)
    plt.title("BC 训练损失"); plt.xlabel("Epoch"); plt.ylabel("平均交叉熵损失"); plt.grid(True); plt.tight_layout()
    bc_plot_path = "bc_loss_curve_optimized.png" # 图的新名称
    plt.savefig(bc_plot_path); plt.close("BC Loss Curve")
    print(f"BC 损失图已保存至: {bc_plot_path}")
    return net

#####################
#   PPO 网络        # (保持不变)
#####################
class PPO(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1) # 输出概率
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1) # 输出单个值
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Actor 提供动作概率，Critic 提供状态值
        return self.actor(x), self.critic(x)
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

#####################
#    PPO Agent      # (保持不变 - update 逻辑自动使用归一化奖励)
#####################
class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.policy = PPO(config.state_dim, config.action_dim, config.hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.initial_learning_rate, eps=1e-5)
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        # 奖励归一化器实例现在是 Agent 的一部分，以正确处理 GAE
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate) if config.normalize_rewards else None
        self.memory: List[Tuple[np.ndarray, int, float, float, bool, np.ndarray]] = [] # (s_raw, a, logp, r_raw, done, next_s_raw)
        self.training_metrics: Dict[str, List[float]] = {"actor_losses": [], "critic_losses": [], "total_losses": [], "entropies": []}

    def load_bc_actor(self, bc_net: Optional[BehaviorCloningNet]):
        if bc_net is None: print("BC net 不可用，跳过权重加载。"); return
        try:
            # 只加载 actor 权重，critic 保持随机初始化
            self.policy.actor.load_state_dict(bc_net.actor.state_dict())
            print("✅ BC Actor 权重已加载到 PPO 策略。")
        except Exception as e: print(f"加载 BC Actor 权重时出错: {e}")

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """使用智能体的归一化器归一化原始状态。"""
        if self.config.normalize_observations and self.obs_normalizer:
            # 仅归一化，此处不更新 (更新发生在 `update` 中)
            norm_state = (state - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8)
            return np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip).astype(np.float32)
        return state.astype(np.float32) # 如果不归一化则返回原始状态

    def get_action(self, raw_state: np.ndarray, current_episode: int) -> Tuple[int, float]:
        """根据带屏蔽的策略概率选择动作。"""
        if not isinstance(raw_state, np.ndarray) or raw_state.shape != (self.config.state_dim,):
             print(f"警告: 在 get_action 中收到无效的 raw_state: {raw_state}。使用零。")
             raw_state = np.zeros(self.config.state_dim)

        # 归一化状态以供策略输入
        normalized_state = self.normalize_state(raw_state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)

        self.policy.eval() # 设置为评估模式以进行确定性动作选择 (使用概率)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
        self.policy.train() # 设置回训练模式

        # --- 动作屏蔽 ---
        # 从原始状态获取当前车道
        try:
             lane_idx = int(round(raw_state[1]))
             num_lanes = self.config.num_train_lanes
             # 以防万一裁剪车道索引
             if not (0 <= lane_idx < num_lanes):
                 lane_idx = np.clip(lane_idx, 0, num_lanes - 1)
        except IndexError:
            print("警告: 为屏蔽访问 raw_state 中的车道时发生 IndexError。假设为车道 0。")
            lane_idx = 0
            num_lanes = self.config.num_train_lanes

        mask = torch.ones_like(probs, dtype=torch.float32)
        if lane_idx == 0: # 如果在最左侧车道
            mask[0, 1] = 0.0 # 禁用动作 1 (左)
        if lane_idx >= (num_lanes - 1): # 如果在最右侧车道
            mask[0, 2] = 0.0 # 禁用动作 2 (右)

        # 应用屏蔽并重新归一化概率
        masked_probs = probs * mask
        probs_sum = masked_probs.sum(dim=-1, keepdim=True)

        # 如果所有有效动作的概率都为零，则避免除以零
        if probs_sum.item() < 1e-8:
            # 如果总和接近零，则默认为 "保持车道" 动作 (0)
            # 创建一个强烈偏向动作 0 的概率分布
            final_probs = torch.zeros_like(probs)
            final_probs[0, 0] = 1.0 # 强制动作 0
            print("警告: 屏蔽后的概率总和为零。默认为动作 0 (保持车道)。")
        else:
            final_probs = masked_probs / probs_sum # 重新归一化

        # 确保 Categorical 分布的概率总和恰好为 1
        final_probs = (final_probs + 1e-9) / (final_probs.sum(dim=-1, keepdim=True) + 1e-9 * self.config.action_dim)

        try:
             # 从最终分布中采样动作
             dist = Categorical(probs=final_probs)
             action = dist.sample()
             log_prob = dist.log_prob(action) # 计算所选动作的对数概率
             return action.item(), log_prob.item()
        except ValueError as e:
             # 如果尽管进行了检查，概率仍然无效，则可能发生这种情况
             print(f"创建 Categorical 分布时出错: {e}。 Probs: {final_probs}")
             # 回退: 返回动作 0 及其对数概率
             action_0_log_prob = torch.log(final_probs[0, 0] + 1e-9) # 动作 0 的对数概率
             return 0, action_0_log_prob.item()

    def store(self, transition: Tuple[np.ndarray, int, float, float, bool, np.ndarray]):
        """存储一个转换 (原始状态, 动作, 对数概率, 原始奖励, 完成标志, 下一个原始状态)。"""
        self.memory.append(transition)

    def update(self, current_episode: int, total_episodes: int):
        """使用收集的内存执行 PPO 更新。"""
        if not self.memory: return 0.0 # 应返回一些内容，可能是平均奖励？更改了返回值。

        # 解包内存 (包含原始状态和原始奖励)
        raw_states = np.array([m[0] for m in self.memory])
        actions = torch.LongTensor(np.array([m[1] for m in self.memory]))
        old_log_probs = torch.FloatTensor(np.array([m[2] for m in self.memory]))
        raw_rewards = np.array([m[3] for m in self.memory]) # 原始奖励
        dones = torch.BoolTensor(np.array([m[4] for m in self.memory]))
        raw_next_states = np.array([m[5] for m in self.memory])

        # --- 观测值归一化 ---
        if self.config.normalize_observations and self.obs_normalizer:
            # 使用这批原始状态更新归一化器统计信息
            self.obs_normalizer.update(raw_states)
            # 使用 *更新后* 的统计信息归一化 states 和 next_states
            states_norm = np.clip((raw_states - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8), -self.config.obs_norm_clip, self.config.obs_norm_clip)
            next_states_norm = np.clip((raw_next_states - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8), -self.config.obs_norm_clip, self.config.obs_norm_clip)
            # 转换为张量
            states = torch.FloatTensor(states_norm).float()
            next_states = torch.FloatTensor(next_states_norm).float()
        else: # 如果归一化关闭，则使用原始状态
            states = torch.FloatTensor(raw_states).float()
            next_states = torch.FloatTensor(raw_next_states).float()

        # --- 奖励归一化 ---
        if self.config.normalize_rewards and self.reward_normalizer:
             # 使用这批原始奖励更新奖励归一化器统计信息
             # 注意: 这基于回报方差估计进行归一化，而不仅仅是奖励
             self.reward_normalizer.update(raw_rewards) # 使用回合回报更新 (可以改进)
             # 归一化内存中的原始奖励
             rewards_normalized = self.reward_normalizer.normalize(raw_rewards, clip=self.config.reward_norm_clip)
             rewards_tensor = torch.FloatTensor(rewards_normalized).float()
        else: # 如果归一化关闭，则使用原始奖励
            rewards_tensor = torch.FloatTensor(raw_rewards).float()

        # --- 计算优势和回报 (GAE) ---
        with torch.no_grad():
            self.policy.eval() # 设置为评估模式以进行价值预测
            values = self.policy.get_value(states).squeeze()
            next_values = self.policy.get_value(next_states).squeeze()
            self.policy.train() # 设置回训练模式

        advantages = torch.zeros_like(rewards_tensor)
        last_gae_lam = 0.0
        num_steps = len(rewards_tensor)

        # GAE 计算 (反向迭代)
        for t in reversed(range(num_steps)):
             # 正确处理终止状态
             is_terminal = dones[t]
             next_val_t = 0.0 if is_terminal else next_values[t] # 如果下一个状态是终止状态，则使用 0

             # 使用可能归一化的奖励计算 TD 误差 (delta)
             delta = rewards_tensor[t] + self.config.gamma * next_val_t - values[t]
             # 计算 GAE 优势
             advantages[t] = last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * (1.0 - is_terminal.float()) * last_gae_lam # 如果是终止状态，则屏蔽未来项

        # 计算回报 (价值函数的目标)
        returns = advantages + values.detach() # 此处使用分离的 values

        # 如果启用，则归一化优势
        if self.config.normalize_advantages:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO 更新循环 ---
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        epoch_actor_losses, epoch_critic_losses, epoch_total_losses, epoch_entropies = [], [], [], []

        # 更新学习率 (线性衰减)
        current_lr = linear_decay(self.config.initial_learning_rate, self.config.final_learning_rate, total_episodes, current_episode)
        for param_group in self.optimizer.param_groups: param_group['lr'] = current_lr

        self.policy.train() # 确保模型处于训练模式

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices) # 每个 epoch 都打乱数据
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                # 获取批量数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx] # Critic 的目标
                batch_old_values = values[batch_idx].detach() # 用于裁剪的旧 values

                # 获取该批次的新策略输出
                new_probs_dist_batch, current_values_batch = self.policy(batch_states)
                current_values_batch = current_values_batch.squeeze()

                # --- 批量更新内的动作屏蔽 (重要!) ---
                # 我们需要与此批次对应的原始状态来获取车道索引
                batch_raw_states = raw_states[batch_idx]
                batch_lanes = torch.tensor(batch_raw_states[:, 1], dtype=torch.long) # 获取车道索引
                num_lanes = self.config.num_train_lanes

                mask_batch = torch.ones_like(new_probs_dist_batch, dtype=torch.float32)
                mask_batch[batch_lanes == 0, 1] = 0.0 # 屏蔽车道 0 的左转
                mask_batch[batch_lanes >= (num_lanes - 1), 2] = 0.0 # 屏蔽最后车道的右转

                masked_new_probs_batch = new_probs_dist_batch * mask_batch
                probs_sum_batch = masked_new_probs_batch.sum(dim=-1, keepdim=True)

                # 处理屏蔽后潜在的零和
                safe_probs_sum_batch = torch.where(probs_sum_batch < 1e-8, torch.ones_like(probs_sum_batch), probs_sum_batch)
                renormalized_probs_batch = masked_new_probs_batch / safe_probs_sum_batch

                # 确保 Categorical 分布的和为 1
                renormalized_probs_batch = (renormalized_probs_batch + 1e-9) / (renormalized_probs_batch.sum(dim=-1, keepdim=True) + 1e-9 * self.config.action_dim)

                try:
                     dist_batch = Categorical(probs=renormalized_probs_batch)
                except ValueError:
                     print(f"警告: PPO 更新批次中存在无效概率。跳过此批次。")
                     continue # 跳过此批次

                # 计算新的对数概率和熵
                new_log_probs_batch = dist_batch.log_prob(batch_actions)
                entropy_batch = dist_batch.entropy().mean() # 该批次的平均熵

                # --- 计算 Actor 损失 (带裁剪的策略梯度) ---
                ratio_batch = torch.exp(new_log_probs_batch - batch_old_log_probs) # 重要性采样比率
                surr1_batch = ratio_batch * batch_advantages
                surr2_batch = torch.clamp(ratio_batch, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                actor_loss_batch = -torch.min(surr1_batch, surr2_batch).mean() # 裁剪目标

                # --- 计算 Critic 损失 (带裁剪的价值函数损失) ---
                if self.config.value_clip:
                    # 根据旧 values 裁剪预测的 values
                    values_pred_clipped_batch = batch_old_values + torch.clamp(current_values_batch - batch_old_values, -self.config.value_clip_epsilon, self.config.value_clip_epsilon)
                    # 计算裁剪和未裁剪预测的 MSE 损失
                    vf_loss1_batch = nn.MSELoss()(current_values_batch, batch_returns)
                    vf_loss2_batch = nn.MSELoss()(values_pred_clipped_batch, batch_returns)
                    # 使用两个损失中的最大值
                    critic_loss_batch = 0.5 * torch.max(vf_loss1_batch, vf_loss2_batch)
                else:
                    # 如果禁用裁剪，则使用标准 MSE 损失
                    critic_loss_batch = 0.5 * nn.MSELoss()(current_values_batch, batch_returns)

                # --- 计算熵损失 ---
                # 获取衰减的熵系数
                current_entropy_coef = linear_decay(self.config.entropy_coef_start, self.config.entropy_coef_end, self.config.entropy_decay_episodes, current_episode) if self.config.use_entropy_decay else self.config.entropy_coef_start
                entropy_loss_batch = -current_entropy_coef * entropy_batch # 最大化熵

                # --- 总损失 ---
                loss_batch = actor_loss_batch + critic_loss_batch + entropy_loss_batch

                # --- 优化步骤 ---
                self.optimizer.zero_grad()
                loss_batch.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.gradient_clip_norm)
                self.optimizer.step()

                # 记录批量指标
                epoch_actor_losses.append(actor_loss_batch.item())
                epoch_critic_losses.append(critic_loss_batch.item())
                epoch_total_losses.append(loss_batch.item())
                epoch_entropies.append(entropy_batch.item())

        # 记录更新步骤的平均指标
        self.training_metrics["actor_losses"].append(np.mean(epoch_actor_losses)); self.training_metrics["critic_losses"].append(np.mean(epoch_critic_losses)); self.training_metrics["total_losses"].append(np.mean(epoch_total_losses)); self.training_metrics["entropies"].append(np.mean(epoch_entropies))

        # 计算此批次的平均原始奖励 (用于潜在日志记录，不用于返回)
        avg_raw_reward = np.mean(raw_rewards) if len(raw_rewards) > 0 else 0.0

        self.memory.clear() # 更新后清除内存

        # 返回已处理批次的平均原始奖励 (可选，可返回 0.0)
        return avg_raw_reward

#####################
#   主训练流程       # (更新绘图名称/目录，通过 SumoEnv 使用优化的奖励)
#####################
def main():
    config = Config()
    # 开始前确保 SUMO 文件存在
    required_files = [config.config_path, "a.net.xml", "a.rou.xml"]
    abort_training = False
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 未找到所需的 SUMO 文件: {f}")
            abort_training = True
    if abort_training:
        print("由于缺少文件，中止训练。")
        sys.exit(1)

    # 设置目录和智能体
    results_dir_base = "." # 保存在当前目录
    os.makedirs(results_dir_base, exist_ok=True)

    env_main = SumoEnv(config) # SumoEnv 现在包含优化的奖励计算
    agent = Agent(config)
    bc_net = None

    # --- 行为克隆阶段 ---
    if config.use_bc:
        print("\n" + "#"*20 + " 阶段 1: 行为克隆 " + "#"*20)
        bc_data = []
        print("\n--- 开始 BC 数据收集 ---")
        for ep in range(config.bc_collect_episodes):
            print(f"BC 数据收集 - 回合 {ep + 1}/{config.bc_collect_episodes}")
            try:
                state_raw = env_main.reset() # 返回原始状态
                done = False; ep_steps = 0
                while not done and ep_steps < config.max_steps:
                    # 在原始状态上使用基于规则的策略
                    action = rule_based_action_improved(state_raw, config)
                    # 步骤环境 (返回下一个原始状态, 原始奖励, 完成标志)
                    next_state_raw, _, done = env_main.step(action)
                    # 如果有效则存储原始状态和动作
                    if not done and not np.any(np.isnan(state_raw)) and not np.any(np.isinf(state_raw)):
                         bc_data.append((state_raw.copy(), action))
                    elif not done: print(f"警告: 在 BC 数据收集回合 {ep+1}, 步骤 {ep_steps} 遇到无效原始状态。跳过。")
                    state_raw = next_state_raw; ep_steps += 1
                print(f"BC 回合 {ep + 1} 完成: {ep_steps} 步。总数据: {len(bc_data)}")
            except (ConnectionError, RuntimeError, traci.exceptions.TraCIException) as e: print(f"\nBC 回合 {ep + 1} 期间出错: {e}"); env_main._close(); time.sleep(1)
            except KeyboardInterrupt: print("\n用户中断 BC 数据收集。"); env_main._close(); return
        print(f"\nBC 数据收集完成。总样本: {len(bc_data)}")
        if bc_data:
             # 将智能体的 obs_normalizer 传递给 bc_train 以使用 BC 数据更新它
             bc_net = bc_train(config, bc_data, agent.obs_normalizer)
             agent.load_bc_actor(bc_net) # 将权重加载到 PPO actor
        else: print("未收集到 BC 数据，跳过训练。")
    else: print("行为克隆已禁用。")

    # --- PPO 训练阶段 ---
    print("\n" + "#"*20 + " 阶段 2: PPO 训练 " + "#"*20)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"ppo_results_optimized_{timestamp}" # 优化运行的新名称
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    print(f"结果将保存在: {results_dir}")
    try:
        config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        config_save_path = os.path.join(results_dir, "config_used.json")
        with open(config_save_path, "w", encoding="utf-8") as f: json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
        print(f"配置已保存至: {config_save_path}")
    except Exception as e: print(f"警告: 无法保存配置 JSON: {e}")

    all_raw_rewards = []; lane_change_counts = []; collision_counts = []; total_steps_per_episode = []; best_avg_reward = -float('inf')

    try:
        for episode in tqdm(range(1, config.ppo_episodes + 1), desc="PPO 训练回合"):
            state_raw = env_main.reset() # 获取初始原始状态
            episode_raw_reward = 0.0; done = False; step_count = 0

            while not done and step_count < config.max_steps:
                # 根据原始状态获取动作和对数概率
                action, log_prob = agent.get_action(state_raw, episode)

                # 步骤环境，获取下一个原始状态、原始奖励、完成标志
                next_state_raw, raw_reward, done = env_main.step(action) # 内部使用优化的奖励计算

                # 存储前验证数据
                valid_data = not (np.any(np.isnan(state_raw)) or np.any(np.isnan(next_state_raw)) or not np.isfinite(raw_reward))

                if valid_data:
                    # 存储原始状态和原始奖励
                    agent.store((state_raw.copy(), action, log_prob, raw_reward, done, next_state_raw.copy()))
                    state_raw = next_state_raw # 更新状态以进行下一次迭代
                    episode_raw_reward += raw_reward # 累积原始奖励以进行日志记录
                else:
                    print(f"警告: 在回合 {episode}, 步骤 {step_count} 从 env.step 获取到无效数据。跳过存储。")
                    # 如果下一个状态无效，则终止回合以避免传播 NaN
                    if np.any(np.isnan(next_state_raw)):
                         done = True
                         print("  -> 由于无效的下一个状态而结束回合。")
                    else:
                         # 如果只有奖励错误，仍然更新状态以尝试恢复
                         state_raw = next_state_raw

                step_count += 1
                if done: break # 如果完成则退出 while 循环

            # 回合结束后 (或达到最大步数) 执行 PPO 更新
            avg_raw_reward_in_batch = agent.update(episode, config.ppo_episodes) # 更新返回批次中的平均原始奖励

            # 记录回合指标
            all_raw_rewards.append(episode_raw_reward)
            lane_change_counts.append(env_main.change_lane_count)
            collision_counts.append(1 if env_main.collision_occurred else 0)
            total_steps_per_episode.append(step_count)

            # 根据原始回合奖励的滚动平均值保存最佳模型
            avg_window = 20
            if episode >= avg_window:
                rewards_slice = all_raw_rewards[max(0, episode - avg_window):episode]
                if rewards_slice:
                   current_avg_reward = np.mean(rewards_slice)
                   if current_avg_reward > best_avg_reward:
                       best_avg_reward = current_avg_reward
                       best_model_path = os.path.join(models_dir, "best_model.pth")
                       torch.save(agent.policy.state_dict(), best_model_path)
                       print(f"\n🎉 新的最佳平均原始奖励 ({avg_window}回合): {best_avg_reward:.2f}! 模型已保存。")

            # 定期日志记录
            if episode % config.log_interval == 0:
                 log_slice_start = max(0, episode - config.log_interval)
                 avg_reward_log = np.mean(all_raw_rewards[log_slice_start:]) if all_raw_rewards[log_slice_start:] else 0
                 avg_steps_log = np.mean(total_steps_per_episode[log_slice_start:]) if total_steps_per_episode[log_slice_start:] else 0
                 collision_rate_log = np.mean(collision_counts[log_slice_start:]) * 100 if collision_counts[log_slice_start:] else 0
                 avg_lc_log = np.mean(lane_change_counts[log_slice_start:]) if lane_change_counts[log_slice_start:] else 0
                 current_lr = agent.optimizer.param_groups[0]['lr']
                 last_entropy = agent.training_metrics['entropies'][-1] if agent.training_metrics['entropies'] else 'N/A'
                 print(f"\n回合: {episode}/{config.ppo_episodes} | 平均原始奖励: {avg_reward_log:.2f} | 最佳平均原始: {best_avg_reward:.2f} | 步数: {avg_steps_log:.1f} | 换道: {avg_lc_log:.1f} | 碰撞: {collision_rate_log:.1f}% | LR: {current_lr:.7f} | 熵: {last_entropy if isinstance(last_entropy, str) else last_entropy:.4f}")

            # 定期模型保存
            if episode % config.save_interval == 0:
                 periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                 torch.save(agent.policy.state_dict(), periodic_model_path)

    except KeyboardInterrupt: print("\n用户中断训练。")
    except Exception as e: print(f"\n训练期间发生致命错误: {e}"); traceback.print_exc()
    finally:
        print("正在关闭最终环境..."); env_main._close()
        if 'agent' in locals() and agent is not None:
            last_model_path = os.path.join(models_dir, "last_model.pth"); torch.save(agent.policy.state_dict(), last_model_path); print(f"最终模型已保存至: {last_model_path}")

            # --- 绘图 (根据要求修改) ---
            print("生成训练图表...")
            plt.figure("PPO 训练曲线 (优化版-修改)", figsize=(12, 6)) # 修改图表名称和大小

            # 图 1: 奖励滚动平均 (来自原 Subplot 1)
            plt.subplot(1, 2, 1)
            plt.grid(True, linestyle='--')
            if len(all_raw_rewards) >= 10:
                rolling_avg_reward = np.convolve(all_raw_rewards, np.ones(10) / 10, mode='valid')
                plt.plot(np.arange(9, len(all_raw_rewards)), rolling_avg_reward, label='10回合滚动平均奖励 (原始值)', color='red', linewidth=2)
            else:
                plt.plot([], label='10回合滚动平均奖励 (数据不足)', color='red', linewidth=2) # 占位符
            plt.title("滚动平均回合奖励")
            plt.xlabel("回合")
            plt.ylabel("平均总奖励 (原始)")
            plt.legend()

            # 图 2: Critic 损失 (来自原 Subplot 4)
            plt.subplot(1, 2, 2)
            plt.grid(True, linestyle='--')
            if agent.training_metrics["critic_losses"]:
                updates_axis = np.arange(len(agent.training_metrics["critic_losses"]))
                plt.plot(updates_axis, agent.training_metrics["critic_losses"], label='Critic 损失', alpha=0.8, color='green') # 只绘制 Critic 损失
                plt.title("每次更新的 PPO Critic 损失")
                plt.xlabel("更新步骤")
                plt.ylabel("Critic 损失")
                plt.legend()
            else:
                 plt.title("PPO Critic 损失 (无数据)")


            plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
            plot_path = os.path.join(results_dir, "training_curves_final_optimized_simple.png") # 新的文件名
            plt.savefig(plot_path);
            plt.close("all");
            print(f"训练图表已保存至: {plot_path}")

            # --- 保存训练数据 ---
            print("正在保存训练数据...");
            # 访问前确保 metrics_per_update 存在
            metrics_data = agent.training_metrics if hasattr(agent, 'training_metrics') else {}
            training_data = {
                "episode_rewards_raw": all_raw_rewards,
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "metrics_per_update": metrics_data # 保存整个字典
                }
            data_path = os.path.join(results_dir, "training_data_final_optimized.json") # 更新的文件名
            try:
                # 修复: 定义自定义序列化器函数，不使用 self
                def default_serializer(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, np.bool_): return bool(obj)
                    # 默认回退，处理无法直接序列化的类型
                    return str(obj)

                # 保存前清理数据 (处理 NaNs/Infs)
                for key, value_list in training_data.items():
                    if isinstance(value_list, dict): # 处理 metrics_per_update 字典
                        for metric_key, metric_list in value_list.items():
                             # 确保列表中的所有项都是有限的浮点数，否则替换为 0.0
                             training_data[key][metric_key] = [0.0 if (m is None or not np.isfinite(m)) else float(m) for m in metric_list]
                    elif isinstance(value_list, list): # 处理列表
                         if key == "episode_rewards_raw":
                              # 确保奖励是有限的浮点数，否则替换为 0.0
                             training_data[key] = [0.0 if (r is None or not np.isfinite(r)) else float(r) for r in value_list]
                         else: # 假设其他列表包含安全类型 (int)
                              # 对列表中的每个项目应用序列化器
                              training_data[key] = [default_serializer(item) for item in value_list]

                with open(data_path, "w", encoding="utf-8") as f:
                     # 在 dump 调用中使用 default 参数
                     json.dump(training_data, f, indent=4, ensure_ascii=False, default=default_serializer)
                print(f"训练数据已保存至: {data_path}")
            except Exception as e: print(f"保存训练数据时出错: {e}"); traceback.print_exc()
        else: print("智能体未初始化，无法保存最终模型/数据。")
        print(f"\n PPO 训练 (优化版) 完成。结果已保存在: {results_dir}")

if __name__ == "__main__":
    main()