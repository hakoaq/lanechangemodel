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
    ppo_episodes = 1000
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
        # Note: This updates based on per-step rewards, not episode returns.
        # For better reward scaling, update with episode returns might be preferred,
        # but this approach scales individual step rewards based on recent reward variance.
        self.returns.extend(rewards) # Add new step/batch rewards
        if len(self.returns) > 1:
             current_mean = np.mean(self.returns)
             current_var = np.var(self.returns)
             # EMA 更新
             self.ret_mean = (1.0 - self.alpha) * self.ret_mean + self.alpha * current_mean
             self.ret_var = (1.0 - self.alpha) * self.ret_var + self.alpha * current_var

    def normalize(self, r: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """使用运行方差归一化奖励。"""
        std = np.sqrt(self.ret_var + self.epsilon)
        # Normalize using the running std deviation of rewards
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

# --- Helper for Plotting Rolling Average ---
def calculate_rolling_average(data, window):
    if len(data) < window:
        return np.array([]) # Not enough data for a full window
    return np.convolve(data, np.ones(window) / window, mode='valid')

#####################
#   SUMO 环境封装    # (保持不变)
#####################
class SumoEnv:
    # __init__, reset_metrics, _start_sumo, _add_ego_vehicle, reset,
    # _get_surrounding_vehicle_info, _get_state, _close
    # step() uses the optimized reward calculation. No changes needed here.

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

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5) # Increased sleep time slightly
                traci.init(self.traci_port)
                # Suppress connection message in training for cleaner logs
                # print(f"✅ SUMO TraCI 已连接 (端口: {self.traci_port}).")
                return
            except traci.exceptions.TraCIException:
                if attempt == connection_attempts - 1:
                    print(f"错误: 达到最大 TraCI 连接尝试次数 (端口: {self.traci_port})。")
                    self._close()
                    raise ConnectionError(f"无法连接到 SUMO (端口: {self.traci_port})。")
            except Exception as e:
                print(f"错误: 连接 TraCI 时发生意外错误: {e}")
                self._close()
                raise ConnectionError(f"连接到 SUMO 时发生未知错误 (端口: {self.traci_port})。")

    def _add_ego_vehicle(self):
        ego_route_id = "route_E0"
        if ego_route_id not in traci.route.getIDList():
             try: traci.route.add(ego_route_id, ["E0"])
             except traci.exceptions.TraCIException as e: raise RuntimeError(f"添加路径 '{ego_route_id}' 失败: {e}")

        if self.config.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.config.ego_type_id)
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0")
                # BC/PPO specific parameters from original config
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcSpeedGain", "1.5")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcCooperative", "0.5")
            except traci.exceptions.TraCIException as e: print(f"警告: 设置 Ego 类型 '{self.config.ego_type_id}' 参数失败: {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            try: traci.vehicle.remove(self.config.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"警告: 移除残留 Ego 失败: {e}")

        try:
            start_lane = np.random.randint(0, self.config.num_train_lanes) # Ensure valid lane
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                              typeID=self.config.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max")
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
                 # Ensure initial lane index is valid
                 initial_lane = int(round(initial_state[1]))
                 self.last_lane_idx = np.clip(initial_lane, 0, self.config.num_train_lanes - 1)
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

        try:
            ego_road = traci.vehicle.getRoadID(ego_id)
            if not ego_road or not ego_road.startswith("E"): return infos

            num_lanes_on_edge = self.config.num_train_lanes

            for veh_id in veh_ids:
                if veh_id == ego_id: continue
                try:
                    if traci.vehicle.getRoadID(veh_id) != ego_road: continue
                    veh_pos = traci.vehicle.getPosition(veh_id)
                    veh_lane = traci.vehicle.getLaneIndex(veh_id)
                    veh_speed = traci.vehicle.getSpeed(veh_id)

                    if not (0 <= veh_lane < num_lanes_on_edge): continue

                    dx = veh_pos[0] - ego_pos[0]; distance = abs(dx)
                    if distance >= max_dist: continue

                    rel_speed = ego_speed - veh_speed

                    if veh_lane == ego_lane:
                        if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                    elif veh_lane == ego_lane - 1:
                        if dx > 0 and distance < infos['left_front'][0]: infos['left_front'] = (distance, rel_speed)
                        elif dx <= 0 and distance < infos['left_back'][0]: infos['left_back'] = (distance, rel_speed)
                    elif veh_lane == ego_lane + 1:
                        if dx > 0 and distance < infos['right_front'][0]: infos['right_front'] = (distance, rel_speed)
                        elif dx <= 0 and distance < infos['right_back'][0]: infos['right_back'] = (distance, rel_speed)
                except traci.exceptions.TraCIException: continue
        except traci.exceptions.TraCIException as e_outer:
            if "Vehicle '" + ego_id + "' is not known" not in str(e_outer):
                print(f"警告: 获取 {ego_id} 周围信息时发生 TraCI 错误: {e_outer}")
            return infos

        return infos


    def _get_state(self) -> np.ndarray:
        """获取原始状态向量"""
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id
        if ego_id not in traci.vehicle.getIDList(): return self.last_raw_state
        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road or not current_road.startswith("E"): return self.last_raw_state

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            num_lanes = self.config.num_train_lanes

            if not (0 <= ego_lane < num_lanes):
                 ego_lane = np.clip(ego_lane, 0, num_lanes - 1)

            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            state[0] = ego_speed
            state[1] = float(ego_lane)
            state[2] = min(surround_info['front'][0], self.config.max_distance)
            state[3] = surround_info['front'][1]
            state[4] = min(surround_info['left_front'][0], self.config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.config.max_distance)
            state[7] = min(surround_info['right_front'][0], self.config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.config.max_distance)
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                 print("警告: 在状态计算中检测到 NaN 或 Inf。使用最后有效状态。")
                 return self.last_raw_state

            self.last_raw_state = state.copy()

        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" not in str(e):
                print(f"警告: 获取 {ego_id} 状态时发生 TraCI 错误: {e}。")
            return self.last_raw_state
        except Exception as e:
            print(f"警告: 获取 {ego_id} 状态时发生未知错误: {e}。"); traceback.print_exc()
            return self.last_raw_state

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行一步，返回 (next_raw_state, raw_reward, done)"""
        done = False; raw_reward = 0.0
        ego_id = self.config.ego_vehicle_id
        self.last_action = action

        if ego_id not in traci.vehicle.getIDList():
             self.collision_occurred = True
             return self.last_raw_state, self.config.reward_collision, True

        try:
            previous_lane_idx = self.last_lane_idx

            current_lane = traci.vehicle.getLaneIndex(ego_id)
            num_lanes = self.config.num_train_lanes
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)

            lane_change_requested = False
            if action == 1 and current_lane > 0:
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
                lane_change_requested = True
            elif action == 2 and current_lane < (num_lanes - 1):
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
                lane_change_requested = True

            traci.simulationStep()
            self.current_step += 1

            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True; done = True
                raw_reward = self.config.reward_collision
                next_state_raw = self.last_raw_state
                # Try to update last_lane_idx based on last known state
                try:
                    self.last_lane_idx = int(round(self.last_raw_state[1])) if len(self.last_raw_state) > 1 else 0
                    self.last_lane_idx = np.clip(self.last_lane_idx, 0, num_lanes - 1)
                except IndexError: self.last_lane_idx = 0
                return next_state_raw, raw_reward, done

            collisions = traci.simulation.getCollisions()
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True; done = True
                    raw_reward = self.config.reward_collision; break

            next_state_raw = self._get_state()
            current_lane_after_step = np.clip(int(round(next_state_raw[1])), 0, num_lanes - 1)

            if not self.collision_occurred:
                current_speed_after_step = next_state_raw[0]
                front_dist_after_step = next_state_raw[2]
                lf_dist_after_step = next_state_raw[4]; lb_dist_after_step = next_state_raw[6]
                rf_dist_after_step = next_state_raw[7]; rb_dist_after_step = next_state_raw[9]
                can_change_left_after_step = next_state_raw[10] > 0.5
                can_change_right_after_step = next_state_raw[11] > 0.5

                actual_lane_change = (previous_lane_idx != current_lane_after_step)
                if actual_lane_change: self.change_lane_count += 1

                raw_reward = self._calculate_reward_optimized(
                    action_taken=action,
                    current_speed=current_speed_after_step,
                    current_lane=current_lane_after_step,
                    front_dist=front_dist_after_step,
                    previous_lane=previous_lane_idx,
                    can_change_left=can_change_left_after_step,
                    can_change_right=can_change_right_after_step,
                    left_front_dist=lf_dist_after_step, left_back_dist=lb_dist_after_step,
                    right_front_dist=rf_dist_after_step, right_back_dist=rb_dist_after_step,
                    lane_change_requested=lane_change_requested
                )

            if traci.simulation.getTime() >= 3600: done = True
            if self.current_step >= self.config.max_steps: done = True

        except traci.exceptions.TraCIException as e:
            # Check if the error is due to the vehicle not being known, which might be expected if it collided/finished
            is_known_error = "Vehicle '" + ego_id + "' is not known" in str(e)
            if not is_known_error: # Log other TraCI errors
                print(f"错误: 在步骤 {self.current_step} 期间发生 TraCI 异常: {e}")
            # Assume collision if TraCI error occurs and vehicle is gone or other error happens
            if is_known_error or ego_id not in traci.vehicle.getIDList(): self.collision_occurred = True
            raw_reward = self.config.reward_collision; done = True
            next_state_raw = self.last_raw_state
        except Exception as e:
            print(f"错误: 在步骤 {self.current_step} 期间发生未知异常: {e}"); traceback.print_exc()
            done = True; raw_reward = self.config.reward_collision
            self.collision_occurred = True; next_state_raw = self.last_raw_state

        self.last_speed = next_state_raw[0]
        # Ensure last lane index is valid before assigning
        try:
            new_lane_idx = int(round(next_state_raw[1]))
            self.last_lane_idx = np.clip(new_lane_idx, 0, self.config.num_train_lanes - 1)
        except IndexError:
            # If next_state_raw is somehow invalid, keep the previous index
            pass

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
        if self.collision_occurred: return 0.0

        try:
            num_lanes = self.config.num_train_lanes
            current_lane = np.clip(current_lane, 0, num_lanes - 1)
            previous_lane = np.clip(previous_lane, 0, num_lanes - 1)

            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            # 1. Speed Reward / Penalty
            speed_diff = abs(current_speed - target_speed)
            speed_reward = np.exp(- (speed_diff / (target_speed * 0.4 + 1e-6))**2 ) * self.config.reward_high_speed_scale
            low_speed_penalty = 0.0
            if current_speed < target_speed * 0.6:
                low_speed_penalty = (current_speed / (target_speed * 0.6) - 1.0) * self.config.reward_low_speed_penalty_scale

            # 2. Safe Distance Penalty (Front)
            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < self.config.max_distance and front_dist < min_safe_dist:
                 safety_dist_penalty = max(-1.0, (front_dist / min_safe_dist - 1.0)) * self.config.safe_distance_penalty_scale

            # 3. Comfort Penalty (Harsh Braking)
            comfort_penalty = 0.0
            acceleration = (current_speed - self.last_speed) / self.config.step_length
            if acceleration < -3.0:
                comfort_penalty = (acceleration + 3.0) * self.config.reward_comfort_penalty_scale # Negative value

            # 4. Time Alive Reward
            time_alive = self.config.time_alive_reward

            # --- Enhanced Lane Change Incentives ---
            lane_change_bonus = 0.0
            staying_slow_penalty = 0.0
            inefficient_change_penalty = 0.0
            fast_lane_preference_reward = 0.0

            actual_lane_change = (current_lane != previous_lane)

            if actual_lane_change:
                previous_max_speed = self.config.lane_max_speeds[previous_lane]
                current_max_speed = self.config.lane_max_speeds[current_lane]
                if current_max_speed > previous_max_speed:
                    lane_change_bonus = self.config.reward_faster_lane_bonus
                elif current_max_speed < previous_max_speed:
                    inefficient_change_penalty = self.config.reward_inefficient_change_penalty

            else: # No actual lane change occurred this step
                # Penalty for not moving to an available faster lane
                left_lane_idx = current_lane - 1
                if left_lane_idx >= 0 and self.config.lane_max_speeds[left_lane_idx] > lane_max_speed:
                    is_left_safe_for_reward = (can_change_left and
                                    left_front_dist > self.config.min_safe_change_dist_front and
                                    left_back_dist > self.config.min_safe_change_dist_back)
                    if is_left_safe_for_reward:
                        # Penalize if agent *could* have safely moved left to faster lane but chose action 0 or 2
                        if action_taken != 1: # If didn't *try* to move left
                           staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale

                right_lane_idx = current_lane + 1
                # Only apply right penalty if left wasn't applicable or chosen
                if staying_slow_penalty == 0.0 and right_lane_idx < num_lanes and self.config.lane_max_speeds[right_lane_idx] > lane_max_speed:
                    is_right_safe_for_reward = (can_change_right and
                                     right_front_dist > self.config.min_safe_change_dist_front and
                                     right_back_dist > self.config.min_safe_change_dist_back)
                    if is_right_safe_for_reward:
                        # Penalize if agent *could* have safely moved right to faster lane but chose action 0 or 1
                        if action_taken != 2: # If didn't *try* to move right
                            staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale


            # Fast Lane Preference Reward
            if num_lanes > 1:
                preference_factor = (num_lanes - 1 - current_lane) / (num_lanes - 1)
                fast_lane_preference_reward = self.config.reward_fast_lane_preference * preference_factor

            total_reward = (speed_reward + low_speed_penalty +
                            safety_dist_penalty + comfort_penalty +
                            time_alive +
                            lane_change_bonus + inefficient_change_penalty + staying_slow_penalty +
                            fast_lane_preference_reward)

            # Add a small penalty specifically if a lane change was requested but didn't happen
            # This might discourage futile attempts if SUMO keeps rejecting them
            # if lane_change_requested and not actual_lane_change:
            #     total_reward -= 0.02 # Small penalty for failed request

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
    # Emergency Maneuver (Imminent collision ahead)
    if front_dist < emergency_dist and front_rel_speed > 2.0: # Front significantly slower
        left_safe_emergency = can_change_left and (lf_dist > required_front_gap * 0.7) and (lb_dist > required_back_gap * 0.7)
        right_safe_emergency = can_change_right and (rf_dist > required_front_gap * 0.7) and (rb_dist > required_back_gap * 0.7)
        if left_safe_emergency: return 1 # Emergency left
        if right_safe_emergency: return 2 # Emergency right
        return 0 # No safe option, stay lane (likely hard brake)
    # Following too closely
    if front_dist < safe_follow_dist and front_rel_speed > 1.0: # Front is slower
        left_safe = can_change_left and (lf_dist > required_front_gap) and (lb_dist > required_back_gap)
        right_safe = can_change_right and (rf_dist > required_front_gap) and (rb_dist > required_back_gap)
        # Prefer changing to potentially faster lane if safe
        prefer_left = left_safe and (left_max_speed > right_max_speed or not right_safe)
        prefer_right = right_safe and (right_max_speed > left_max_speed or not left_safe)
        if prefer_left: return 1
        if prefer_right: return 2
        # If speeds similar or only one option safe, take the safe one
        if left_safe: return 1
        if right_safe: return 2
        return 0 # No safe change, stay lane
    # Opportunity to move to a faster lane (Speed Gain)
    speed_threshold = current_max_speed * 0.85 # Consider change if below 85% of lane max
    if ego_speed < speed_threshold:
        # Check left if significantly faster
        if can_change_left and left_max_speed > current_max_speed * 1.05: # Left >5% faster
            left_extra_safe = (lf_dist > required_front_gap * 1.5) and (lb_dist > required_back_gap * 1.2) # Need larger gap
            if left_extra_safe: return 1
        # Check right if significantly faster
        if can_change_right and right_max_speed > current_max_speed * 1.05: # Right >5% faster
            right_extra_safe = (rf_dist > required_front_gap * 1.5) and (rb_dist > required_back_gap * 1.2) # Need larger gap
            if right_extra_safe: return 2
    # Default: Keep Lane
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
        # If enabled, apply normalization using the provided normalizer
        if config.normalize_observations and obs_normalizer:
             # Update normalizer with BC data
             obs_normalizer.update(states_raw)
             # Normalize states for training
             states_normalized = (states_raw - obs_normalizer.mean) / (obs_normalizer.std + 1e-8)
             states_normalized = np.clip(states_normalized, -config.obs_norm_clip, config.obs_norm_clip)
             states_tensor = torch.FloatTensor(states_normalized)
             print("BC 数据已使用运行统计信息进行归一化。")
        else: states_tensor = torch.FloatTensor(states_raw) # Use raw states if normalization off
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
    # Save loss curve plot
    plt.figure("BC Loss Curve", figsize=(8, 4)); plt.plot(all_bc_losses)
    plt.title("BC 训练损失"); plt.xlabel("Epoch"); plt.ylabel("平均交叉熵损失"); plt.grid(True); plt.tight_layout()
    bc_plot_path = "bc_loss_curve_optimized.png" # New name for the plot
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
            nn.Linear(hidden_size, action_dim), nn.Softmax(dim=-1) # Output probabilities
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1) # Output single value
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Actor provides action probabilities, Critic provides state value
        return self.actor(x), self.critic(x)
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

#####################
#    PPO Agent      # (保持不变 - update logic uses normalized rewards internally if enabled)
#####################
class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.policy = PPO(config.state_dim, config.action_dim, config.hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.initial_learning_rate, eps=1e-5)
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        # Reward normalizer instance is now part of Agent to handle GAE correctly
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate) if config.normalize_rewards else None
        self.memory: List[Tuple[np.ndarray, int, float, float, bool, np.ndarray]] = [] # (s_raw, a, logp, r_raw, done, next_s_raw)
        self.training_metrics: Dict[str, List[float]] = collections.defaultdict(list) # Use defaultdict

    def load_bc_actor(self, bc_net: Optional[BehaviorCloningNet]):
        if bc_net is None: print("BC net 不可用，跳过权重加载。"); return
        try:
            # Load only actor weights, critic remains randomly initialized
            self.policy.actor.load_state_dict(bc_net.actor.state_dict())
            print("✅ BC Actor 权重已加载到 PPO 策略。")
        except Exception as e: print(f"加载 BC Actor 权重时出错: {e}")

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize raw state using the agent's normalizer."""
        if self.config.normalize_observations and self.obs_normalizer:
            # Only normalize, do not update here (update happens in `update`)
            norm_state = (state - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8)
            return np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip).astype(np.float32)
        return state.astype(np.float32) # Return raw state if not normalizing

    def get_action(self, raw_state: np.ndarray, current_episode: int) -> Tuple[int, float]:
        """Select action based on policy probabilities with masking."""
        if not isinstance(raw_state, np.ndarray) or raw_state.shape != (self.config.state_dim,):
             print(f"警告: 在 get_action 中收到无效的 raw_state: {raw_state}。使用零。")
             raw_state = np.zeros(self.config.state_dim)

        # Normalize state for policy input
        normalized_state = self.normalize_state(raw_state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)

        self.policy.eval() # Set to eval mode for deterministic action selection (using probs)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
        self.policy.train() # Set back to train mode

        # --- Action Masking ---
        try:
             lane_idx = int(round(raw_state[1]))
             num_lanes = self.config.num_train_lanes
             if not (0 <= lane_idx < num_lanes):
                 lane_idx = np.clip(lane_idx, 0, num_lanes - 1)
        except IndexError:
            print("警告: 为屏蔽访问 raw_state 中的车道时发生 IndexError。假设为车道 0。")
            lane_idx = 0; num_lanes = self.config.num_train_lanes

        mask = torch.ones_like(probs, dtype=torch.float32)
        if lane_idx == 0: mask[0, 1] = 0.0 # Disable action 1 (Left)
        if lane_idx >= (num_lanes - 1): mask[0, 2] = 0.0 # Disable action 2 (Right)

        masked_probs = probs * mask
        probs_sum = masked_probs.sum(dim=-1, keepdim=True)

        if probs_sum.item() < 1e-8:
            final_probs = torch.zeros_like(probs); final_probs[0, 0] = 1.0
            # print("警告: 屏蔽后的概率总和为零。默认为动作 0 (保持车道)。") # Reduce log spam
        else:
            final_probs = masked_probs / probs_sum

        final_probs = (final_probs + 1e-9) / (final_probs.sum(dim=-1, keepdim=True) + 1e-9 * self.config.action_dim)

        try:
             dist = Categorical(probs=final_probs)
             action = dist.sample()
             log_prob = dist.log_prob(action)
             return action.item(), log_prob.item()
        except ValueError as e:
             print(f"创建 Categorical 分布时出错: {e}。 Probs: {final_probs}")
             action_0_log_prob = torch.log(final_probs[0, 0] + 1e-9)
             return 0, action_0_log_prob.item()

    def store(self, transition: Tuple[np.ndarray, int, float, float, bool, np.ndarray]):
        """Store a transition (raw_state, action, log_prob, raw_reward, done, next_raw_state)."""
        self.memory.append(transition)

    def update(self, current_episode: int, total_episodes: int):
        """Perform PPO update using the collected memory."""
        if not self.memory: return 0.0

        # Unpack memory (contains raw states and raw rewards)
        raw_states = np.array([m[0] for m in self.memory])
        actions = torch.LongTensor(np.array([m[1] for m in self.memory]))
        old_log_probs = torch.FloatTensor(np.array([m[2] for m in self.memory]))
        raw_rewards = np.array([m[3] for m in self.memory]) # Raw rewards
        dones = torch.BoolTensor(np.array([m[4] for m in self.memory]))
        raw_next_states = np.array([m[5] for m in self.memory])

        # --- Observation Normalization ---
        if self.config.normalize_observations and self.obs_normalizer:
            self.obs_normalizer.update(raw_states) # Update normalizer stats
            states_norm = np.clip((raw_states - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8), -self.config.obs_norm_clip, self.config.obs_norm_clip)
            next_states_norm = np.clip((raw_next_states - self.obs_normalizer.mean) / (self.obs_normalizer.std + 1e-8), -self.config.obs_norm_clip, self.config.obs_norm_clip)
            states = torch.FloatTensor(states_norm).float()
            next_states = torch.FloatTensor(next_states_norm).float()
        else:
            states = torch.FloatTensor(raw_states).float()
            next_states = torch.FloatTensor(raw_next_states).float()

        # --- Reward Normalization ---
        if self.config.normalize_rewards and self.reward_normalizer:
             # Update reward normalizer stats using these raw rewards
             self.reward_normalizer.update(raw_rewards)
             rewards_normalized = self.reward_normalizer.normalize(raw_rewards, clip=self.config.reward_norm_clip)
             rewards_tensor = torch.FloatTensor(rewards_normalized).float()
        else:
            rewards_tensor = torch.FloatTensor(raw_rewards).float()

        # --- Calculate Advantages and Returns (GAE) ---
        with torch.no_grad():
            self.policy.eval(); values = self.policy.get_value(states).squeeze(); next_values = self.policy.get_value(next_states).squeeze(); self.policy.train()

        advantages = torch.zeros_like(rewards_tensor); last_gae_lam = 0.0; num_steps = len(rewards_tensor)
        for t in reversed(range(num_steps)):
             is_terminal = dones[t]; next_val_t = 0.0 if is_terminal else next_values[t]
             delta = rewards_tensor[t] + self.config.gamma * next_val_t - values[t]
             advantages[t] = last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * (1.0 - is_terminal.float()) * last_gae_lam
        returns = advantages + values.detach()

        if self.config.normalize_advantages:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO Update Loop ---
        dataset_size = states.size(0); indices = np.arange(dataset_size)
        epoch_actor_losses, epoch_critic_losses, epoch_total_losses, epoch_entropies = [], [], [], []

        current_lr = linear_decay(self.config.initial_learning_rate, self.config.final_learning_rate, total_episodes, current_episode)
        for param_group in self.optimizer.param_groups: param_group['lr'] = current_lr

        self.policy.train()
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size; batch_idx = indices[start:end]

                batch_states = states[batch_idx]; batch_actions = actions[batch_idx]; batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]; batch_returns = returns[batch_idx]; batch_old_values = values[batch_idx].detach()

                new_probs_dist_batch, current_values_batch = self.policy(batch_states); current_values_batch = current_values_batch.squeeze()

                # --- Action Masking within Batch Update ---
                batch_raw_states = raw_states[batch_idx]; batch_lanes = torch.tensor(batch_raw_states[:, 1], dtype=torch.long)
                num_lanes = self.config.num_train_lanes
                mask_batch = torch.ones_like(new_probs_dist_batch, dtype=torch.float32)
                mask_batch[batch_lanes == 0, 1] = 0.0
                mask_batch[batch_lanes >= (num_lanes - 1), 2] = 0.0
                masked_new_probs_batch = new_probs_dist_batch * mask_batch
                probs_sum_batch = masked_new_probs_batch.sum(dim=-1, keepdim=True)
                safe_probs_sum_batch = torch.where(probs_sum_batch < 1e-8, torch.ones_like(probs_sum_batch), probs_sum_batch)
                renormalized_probs_batch = masked_new_probs_batch / safe_probs_sum_batch
                renormalized_probs_batch = (renormalized_probs_batch + 1e-9) / (renormalized_probs_batch.sum(dim=-1, keepdim=True) + 1e-9 * self.config.action_dim)

                try: dist_batch = Categorical(probs=renormalized_probs_batch)
                except ValueError: print(f"警告: PPO 更新批次中存在无效概率。跳过此批次。"); continue

                new_log_probs_batch = dist_batch.log_prob(batch_actions); entropy_batch = dist_batch.entropy().mean()

                # --- Actor Loss ---
                ratio_batch = torch.exp(new_log_probs_batch - batch_old_log_probs)
                surr1_batch = ratio_batch * batch_advantages
                surr2_batch = torch.clamp(ratio_batch, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                actor_loss_batch = -torch.min(surr1_batch, surr2_batch).mean()

                # --- Critic Loss ---
                if self.config.value_clip:
                    values_pred_clipped_batch = batch_old_values + torch.clamp(current_values_batch - batch_old_values, -self.config.value_clip_epsilon, self.config.value_clip_epsilon)
                    vf_loss1_batch = nn.MSELoss()(current_values_batch, batch_returns); vf_loss2_batch = nn.MSELoss()(values_pred_clipped_batch, batch_returns)
                    critic_loss_batch = 0.5 * torch.max(vf_loss1_batch, vf_loss2_batch)
                else:
                    critic_loss_batch = 0.5 * nn.MSELoss()(current_values_batch, batch_returns)

                # --- Entropy Loss ---
                current_entropy_coef = linear_decay(self.config.entropy_coef_start, self.config.entropy_coef_end, self.config.entropy_decay_episodes, current_episode) if self.config.use_entropy_decay else self.config.entropy_coef_start
                entropy_loss_batch = -current_entropy_coef * entropy_batch

                # --- Total Loss and Optimization ---
                loss_batch = actor_loss_batch + critic_loss_batch + entropy_loss_batch
                self.optimizer.zero_grad(); loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.gradient_clip_norm); self.optimizer.step()

                epoch_actor_losses.append(actor_loss_batch.item()); epoch_critic_losses.append(critic_loss_batch.item()); epoch_total_losses.append(loss_batch.item()); epoch_entropies.append(entropy_batch.item())

        # Record average metrics for this update step
        self.training_metrics["actor_losses"].append(np.mean(epoch_actor_losses)); self.training_metrics["critic_losses"].append(np.mean(epoch_critic_losses)); self.training_metrics["total_losses"].append(np.mean(epoch_total_losses)); self.training_metrics["entropies"].append(np.mean(epoch_entropies))

        avg_raw_reward = np.mean(raw_rewards) if len(raw_rewards) > 0 else 0.0
        self.memory.clear()
        return avg_raw_reward


#####################
#   主训练流程       #
#####################
def main():
    config = Config()
    # --- File Checks ---
    required_files = [config.config_path, "a.net.xml", "a.rou.xml"]
    abort_training = False
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 未找到所需的 SUMO 文件: {f}")
            abort_training = True
    if abort_training: sys.exit(1)

    # --- Setup ---
    env_main = SumoEnv(config)
    agent = Agent(config)
    bc_net = None
    learning_rates = [] # Store LR per episode for plotting

    # --- Behavior Cloning Phase ---
    if config.use_bc:
        print("\n" + "#"*20 + " 阶段 1: 行为克隆 " + "#"*20)
        bc_data = []
        print("\n--- 开始 BC 数据收集 ---")
        for ep in range(config.bc_collect_episodes):
            print(f"BC 数据收集 - 回合 {ep + 1}/{config.bc_collect_episodes}")
            try:
                state_raw = env_main.reset()
                done = False; ep_steps = 0
                while not done and ep_steps < config.max_steps:
                    action = rule_based_action_improved(state_raw, config)
                    next_state_raw, _, done = env_main.step(action)
                    if not done and not np.any(np.isnan(state_raw)) and not np.any(np.isinf(state_raw)):
                         bc_data.append((state_raw.copy(), action))
                    # elif not done: print(f"警告: 在 BC 数据收集回合 {ep+1}, 步骤 {ep_steps} 遇到无效原始状态。跳过。") # Reduce spam
                    state_raw = next_state_raw; ep_steps += 1
                # print(f"BC 回合 {ep + 1} 完成: {ep_steps} 步。总数据: {len(bc_data)}") # Reduce spam
            except (ConnectionError, RuntimeError, traci.exceptions.TraCIException) as e: print(f"\nBC 回合 {ep + 1} 期间出错: {e}"); env_main._close(); time.sleep(1)
            except KeyboardInterrupt: print("\n用户中断 BC 数据收集。"); env_main._close(); return
        print(f"\nBC 数据收集完成。总样本: {len(bc_data)}")
        if bc_data:
             bc_net = bc_train(config, bc_data, agent.obs_normalizer)
             agent.load_bc_actor(bc_net)
        else: print("未收集到 BC 数据，跳过训练。")
    else: print("行为克隆已禁用。")

    # --- PPO Training Phase ---
    print("\n" + "#"*20 + " 阶段 2: PPO 训练 " + "#"*20)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"ppo_results_optimized_{timestamp}" # New name for optimized run
    models_dir = os.path.join(results_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    print(f"结果将保存在: {results_dir}")
    try: # Save config
        config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        config_save_path = os.path.join(results_dir, "config_used.json")
        with open(config_save_path, "w", encoding="utf-8") as f: json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e: print(f"警告: 无法保存配置 JSON: {e}")

    all_raw_rewards = []; lane_change_counts = []; collision_counts = []; total_steps_per_episode = []; best_avg_reward = -float('inf')

    try:
        for episode in tqdm(range(1, config.ppo_episodes + 1), desc="PPO 训练回合"):
            state_raw = env_main.reset()
            episode_raw_reward = 0.0; done = False; step_count = 0
            # Store current learning rate for this episode
            current_lr = agent.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            while not done and step_count < config.max_steps:
                action, log_prob = agent.get_action(state_raw, episode)
                next_state_raw, raw_reward, done = env_main.step(action)

                valid_data = not (np.any(np.isnan(state_raw)) or np.any(np.isnan(next_state_raw)) or not np.isfinite(raw_reward))
                if valid_data:
                    agent.store((state_raw.copy(), action, log_prob, raw_reward, done, next_state_raw.copy()))
                    state_raw = next_state_raw
                    episode_raw_reward += raw_reward
                else:
                    # print(f"警告: 在回合 {episode}, 步骤 {step_count} 从 env.step 获取到无效数据。跳过存储。") # Reduce spam
                    if np.any(np.isnan(next_state_raw)): done = True
                    else: state_raw = next_state_raw

                step_count += 1
                if done: break

            agent.update(episode, config.ppo_episodes)

            all_raw_rewards.append(episode_raw_reward)
            lane_change_counts.append(env_main.change_lane_count)
            collision_counts.append(1 if env_main.collision_occurred else 0)
            total_steps_per_episode.append(step_count)

            # Save best model based on rolling raw reward
            avg_window = 20
            if episode >= avg_window:
                rewards_slice = all_raw_rewards[max(0, episode - avg_window):episode]
                current_avg_reward = np.mean(rewards_slice) if rewards_slice else -float('inf')
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    best_model_path = os.path.join(models_dir, "best_model.pth")
                    torch.save(agent.policy.state_dict(), best_model_path)
                    print(f"\n🎉 新的最佳平均原始奖励 ({avg_window}回合): {best_avg_reward:.2f}! 模型已保存。")

            if episode % config.log_interval == 0:
                 log_slice_start = max(0, episode - config.log_interval)
                 avg_reward_log = np.mean(all_raw_rewards[log_slice_start:]) if all_raw_rewards[log_slice_start:] else 0
                 avg_steps_log = np.mean(total_steps_per_episode[log_slice_start:]) if total_steps_per_episode[log_slice_start:] else 0
                 collision_rate_log = np.mean(collision_counts[log_slice_start:]) * 100 if collision_counts[log_slice_start:] else 0
                 avg_lc_log = np.mean(lane_change_counts[log_slice_start:]) if lane_change_counts[log_slice_start:] else 0
                 last_entropy = agent.training_metrics['entropies'][-1] if agent.training_metrics['entropies'] else 'N/A'
                 entropy_str = f"{last_entropy:.4f}" if isinstance(last_entropy, (float, np.floating)) else last_entropy
                 print(f"\n回合: {episode}/{config.ppo_episodes} | 平均原始奖励: {avg_reward_log:.2f} | 最佳平均原始: {best_avg_reward:.2f} | 步数: {avg_steps_log:.1f} | 换道: {avg_lc_log:.1f} | 碰撞: {collision_rate_log:.1f}% | LR: {current_lr:.7f} | 熵: {entropy_str}")

            if episode % config.save_interval == 0:
                 periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                 torch.save(agent.policy.state_dict(), periodic_model_path)

    except KeyboardInterrupt: print("\n用户中断训练。")
    except Exception as e: print(f"\n训练期间发生致命错误: {e}"); traceback.print_exc()
    finally:
        print("正在关闭最终环境..."); env_main._close()
        if 'agent' in locals() and agent is not None:
            last_model_path = os.path.join(models_dir, "last_model.pth"); torch.save(agent.policy.state_dict(), last_model_path); print(f"最终模型已保存至: {last_model_path}")

            # --- Plotting (Extended) ---
            print("\n--- 生成训练图表 ---")
            plot_window = 20 # Rolling average window for plots

            # --- Plot 1: Rewards and Steps ---
            print("正在生成奖励和步数图...")
            fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig1.suptitle("PPO 训练: 奖励和回合步数 (优化版)", fontsize=16)

            # Subplot 1.1: Rolling Average Reward
            ax = axs1[0]
            ax.set_ylabel("平均总奖励 (原始)")
            ax.grid(True, linestyle='--')
            if len(all_raw_rewards) >= plot_window:
                rolling_avg_reward = calculate_rolling_average(all_raw_rewards, plot_window)
                episode_axis_rolled = np.arange(plot_window - 1, len(all_raw_rewards))
                ax.plot(episode_axis_rolled, rolling_avg_reward, label=f'{plot_window}回合滚动平均奖励', color='red', linewidth=2)
            else:
                ax.plot([], label=f'{plot_window}回合滚动平均奖励 (数据不足)', color='red', linewidth=2)
            ax.legend()

            # Subplot 1.2: Rolling Average Steps
            ax = axs1[1]
            ax.set_xlabel("回合")
            ax.set_ylabel("平均回合步数")
            ax.grid(True, linestyle='--')
            if len(total_steps_per_episode) >= plot_window:
                 rolling_avg_steps = calculate_rolling_average(total_steps_per_episode, plot_window)
                 episode_axis_rolled = np.arange(plot_window - 1, len(total_steps_per_episode))
                 ax.plot(episode_axis_rolled, rolling_avg_steps, label=f'{plot_window}回合滚动平均步数', color='blue', linewidth=2)
            else:
                 ax.plot([], label=f'{plot_window}回合滚动平均步数 (数据不足)', color='blue', linewidth=2)
            ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path1 = os.path.join(results_dir, "training_rewards_steps.png")
            plt.savefig(plot_path1)
            plt.close(fig1)
            print(f"奖励/步数图已保存至: {plot_path1}")


            # --- Plot 2: Losses ---
            print("正在生成损失图...")
            fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig2.suptitle("PPO 训练: 损失函数 (优化版)", fontsize=16)
            update_axis = np.arange(len(agent.training_metrics.get("actor_losses", [])))

            # Subplot 2.1: Actor Loss
            ax = axs2[0]
            ax.set_ylabel("Actor 损失")
            ax.grid(True, linestyle='--')
            if update_axis.size > 0:
                ax.plot(update_axis, agent.training_metrics.get("actor_losses", []), label='Actor 损失 (PPO Clip)', color='purple', alpha=0.8)
            ax.legend()

            # Subplot 2.2: Critic Loss
            ax = axs2[1]
            ax.set_xlabel("更新步骤")
            ax.set_ylabel("Critic 损失 (MSE)")
            ax.grid(True, linestyle='--')
            if update_axis.size > 0:
                ax.plot(update_axis, agent.training_metrics.get("critic_losses", []), label='Critic 损失', color='green', alpha=0.8)
            ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path2 = os.path.join(results_dir, "training_losses.png")
            plt.savefig(plot_path2)
            plt.close(fig2)
            print(f"损失图已保存至: {plot_path2}")

            # --- Plot 3: Entropy and Learning Rate ---
            print("正在生成熵和学习率图...")
            fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # Separate X-axis
            fig3.suptitle("PPO 训练: 熵和学习率 (优化版)", fontsize=16)

            # Subplot 3.1: Entropy
            ax = axs3[0]
            ax.set_xlabel("更新步骤")
            ax.set_ylabel("策略熵")
            ax.grid(True, linestyle='--')
            update_axis_entropy = np.arange(len(agent.training_metrics.get("entropies", [])))
            if update_axis_entropy.size > 0:
                ax.plot(update_axis_entropy, agent.training_metrics.get("entropies", []), label='策略熵', color='cyan', alpha=0.9)
            ax.legend()

            # Subplot 3.2: Learning Rate
            ax = axs3[1]
            ax.set_xlabel("回合")
            ax.set_ylabel("学习率")
            ax.grid(True, linestyle='--')
            episode_axis_lr = np.arange(len(learning_rates))
            if episode_axis_lr.size > 0:
                ax.plot(episode_axis_lr, learning_rates, label='学习率衰减', color='magenta')
            ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path3 = os.path.join(results_dir, "training_entropy_lr.png")
            plt.savefig(plot_path3)
            plt.close(fig3)
            print(f"熵/学习率图已保存至: {plot_path3}")

            # --- Plot 4: Collisions and Lane Changes ---
            print("正在生成碰撞和换道图...")
            fig4, axs4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig4.suptitle("PPO 训练: 碰撞率和换道次数 (优化版)", fontsize=16)

            # Subplot 4.1: Collision Rate
            ax = axs4[0]
            ax.set_ylabel("碰撞率 (%)")
            ax.grid(True, linestyle='--')
            ax.set_ylim(-5, 105) # Set Y axis from 0 to 100%
            if len(collision_counts) >= plot_window:
                collision_rate = np.array(collision_counts) * 100
                rolling_avg_coll = calculate_rolling_average(collision_rate, plot_window)
                episode_axis_rolled_coll = np.arange(plot_window - 1, len(collision_counts))
                ax.plot(episode_axis_rolled_coll, rolling_avg_coll, label=f'{plot_window}回合滚动平均碰撞率', color='black', linewidth=2)
            else:
                ax.plot([], label=f'{plot_window}回合滚动平均碰撞率 (数据不足)', color='black', linewidth=2)
            ax.legend()

            # Subplot 4.2: Lane Changes
            ax = axs4[1]
            ax.set_xlabel("回合")
            ax.set_ylabel("平均换道次数")
            ax.grid(True, linestyle='--')
            if len(lane_change_counts) >= plot_window:
                rolling_avg_lc = calculate_rolling_average(lane_change_counts, plot_window)
                episode_axis_rolled_lc = np.arange(plot_window - 1, len(lane_change_counts))
                ax.plot(episode_axis_rolled_lc, rolling_avg_lc, label=f'{plot_window}回合滚动平均换道次数', color='orange', linewidth=2)
            else:
                ax.plot([], label=f'{plot_window}回合滚动平均换道次数 (数据不足)', color='orange', linewidth=2)
            ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path4 = os.path.join(results_dir, "training_collisions_lc.png")
            plt.savefig(plot_path4)
            plt.close(fig4)
            print(f"碰撞/换道图已保存至: {plot_path4}")


            # --- Save Training Data ---
            print("正在保存训练数据...");
            training_data = {
                "episode_rewards_raw": all_raw_rewards,
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "metrics_per_update": dict(agent.training_metrics), # Convert defaultdict to dict
                "learning_rates_per_episode": learning_rates
                }
            data_path = os.path.join(results_dir, "training_data_final_optimized.json")
            try:
                def default_serializer(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, np.bool_): return bool(obj)
                    return str(obj)

                # Clean data before saving
                for key, value_list in training_data.items():
                    if isinstance(value_list, dict):
                        for metric_key, metric_list in value_list.items():
                             training_data[key][metric_key] = [0.0 if (m is None or not np.isfinite(m)) else float(m) for m in metric_list]
                    elif isinstance(value_list, list):
                         if key == "episode_rewards_raw":
                             training_data[key] = [0.0 if (r is None or not np.isfinite(r)) else float(r) for r in value_list]
                         else:
                              training_data[key] = [default_serializer(item) for item in value_list]

                with open(data_path, "w", encoding="utf-8") as f:
                     json.dump(training_data, f, indent=4, ensure_ascii=False, default=default_serializer)
                print(f"训练数据已保存至: {data_path}")
            except Exception as e: print(f"保存训练数据时出错: {e}"); traceback.print_exc()
        else: print("智能体未初始化，无法保存最终模型/数据。")
        print(f"\n PPO 训练 (优化版) 完成。结果已保存在: {results_dir}")

if __name__ == "__main__":
    main()