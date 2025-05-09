# -*- coding: utf-8 -*-
"""
智能汽车安全变道决策系统的建模与仿真 - 深度强化学习 (DQN) 训练代码 (优化版 V2)

本代码实现了基于深度 Q 网络 (DQN) 的智能汽车安全变道决策模型，
并在 SUMO 交通仿真环境中进行训练。

主要改进包括：
-   使用更稳定的 Huber 损失 (注: 对于C51, 实际使用交叉熵损失)
-   动态目标网络更新
-   改进的探索策略 (Noisy Nets)
-   优先经验回放 (PER)
-   N 步回报
-   分布式 DQN (C51)
-   状态和奖励归一化
-   详细的日志记录和绘图
-   更健壮的错误处理和异常捕获
-   代码结构优化和模块化
-   兼容CPU和GPU
-   中文注释和文档

V2 Optimizations:
-   Learning Rate Scheduling (Linear Decay)
-   Refined reward parameters for lane choice encouragement
-   Recommendation for longer training duration
"""
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR # For Learning Rate Scheduling
import traci
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple
import socket
import traceback
import math
import logging
import typing

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#####################
#     配置区域       #
#####################
class Config:
    """
    配置类，用于存储所有超参数和设置。
    """
    # --- SUMO 配置 ---
    sumo_binary = "sumo"  # or "sumo-gui"
    config_path = "a.sumocfg"
    step_length = 0.2
    ego_vehicle_id = "drl_ego_car"
    ego_type_id = "car_ego"
    port_range = (8890, 8900)

    # --- DQN 训练 ---
    # 重要提示: 对于复杂任务和深度强化学习, 1000 回合可能不足以完全收敛。
    # 考虑增加到 2000-5000+ 回合或更多，并监控学习曲线。
    dqn_episodes = 1000  # 训练回合数
    max_steps = 8000  # 每回合最大步数
    log_interval = 10  # 日志记录间隔
    save_interval = 50  # 模型保存间隔

    # --- DQN 超参数 ---
    gamma = 0.99  # 折扣因子
    initial_learning_rate = 7e-5
    # 学习率调度参数
    lr_decay_total_updates = int((dqn_episodes * 250 - 7500) * 0.8) # 预估总更新次数的80%用于衰减 (250为预估每回合平均步数)
                                                                  # 如果学习开始步数或回合数变化，需调整此值
    lr_end_factor = 0.1 # 学习率衰减到的最终比例 (e.g., 0.1 * initial_learning_rate)


    batch_size = 512  # 批量大小
    hidden_size = 256  # 隐藏层大小
    replay_buffer_size = 100000  # 回放缓冲区大小
    target_update_freq = 2500  # 目标网络更新频率 (步数)
    learning_starts = 7500  # 开始学习前收集的步数

    use_double_dqn = True  # 使用 Double DQN

    # --- Noisy Networks ---
    use_noisy_nets = True  # 启用 Noisy Networks
    noisy_sigma_init = 0.5  # NoisyLinear 层的初始标准差

    # --- Distributional DQN (C51) ---
    use_distributional = True  # 启用 C51
    v_min = -110  # 最小回报值 (根据奖励调整)
    v_max = 30  # 最大回报值 (根据奖励调整)
    num_atoms = 51  # 分布原子数

    # --- 归一化 ---
    normalize_observations = True  # 归一化观测值
    normalize_rewards = True  # 归一化奖励
    obs_norm_clip = 5.0  # 观测值裁剪范围
    reward_norm_clip = 10.0  # 奖励裁剪范围
    norm_update_rate = 0.001  # 归一化更新速率 (EMA alpha)

    # --- Prioritized Experience Replay (PER) ---
    use_per = True
    per_alpha = 0.6  # 优先级指数
    per_beta_start = 0.4  # 初始重要性采样权重
    per_beta_end = 1.0  # 最终重要性采样权重
    per_beta_annealing_steps = int(dqn_episodes * 300 * 0.8)  # Beta 退火步数 (300为预估每回合平均步数)
    per_epsilon = 1e-5  # 避免除零的小常数

    # --- N-Step Returns ---
    use_n_step = True
    n_step = 5  # N 步回报的步数

    # --- 状态/动作空间 ---
    state_dim = 12
    action_dim = 3  # 0: 保持, 1: 左转, 2: 右转

    # --- 环境参数 ---
    max_speed_global = 33.33  # m/s (~120 km/h)
    max_distance = 100.0  # m
    lane_max_speeds = [33.33, 27.78, 22.22]  # m/s - 必须与 a.net.xml 匹配
    num_train_lanes = len(lane_max_speeds)

    # --- 奖励函数参数 (优化调整) ---
    reward_collision = -100.0
    reward_high_speed_scale = 0.25
    reward_low_speed_penalty_scale = 0.1
    reward_lane_change_penalty = -0.02
    reward_faster_lane_bonus = 0.8  # V2: 增加 (原为 0.6) - 更强激励换到更快车道
    reward_staying_slow_penalty_scale = 0.15 # V2: 增加 (原为 0.1) - 更强惩罚在有更快选择时保持慢车道
    time_alive_reward = 0.01
    reward_comfort_penalty_scale = 0.05
    target_speed_factor = 0.95
    safe_distance_penalty_scale = 0.2
    min_buffer_dist_reward = 5.0
    time_gap_reward = 0.8
    min_safe_change_dist = 15.0


#####################
#   归一化          #
#####################
class RunningMeanStd:
    """
    计算运行均值和标准差，用于状态归一化。
    """
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4, alpha: float = 0.001):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.alpha = alpha

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1.0 - self.alpha) * self.var + self.alpha * batch_var

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-8))


class RewardNormalizer:
    """
    归一化奖励。
    """
    def __init__(self, gamma: float, epsilon: float = 1e-8, alpha: float = 0.001, clip: float = 10.0):
        self.ret_rms = RunningMeanStd(shape=(), alpha=alpha)
        self.epsilon = epsilon
        self.clip = clip

    def normalize(self, r: np.ndarray) -> np.ndarray:
        if not isinstance(r, np.ndarray):
            r = np.array(r)
        if r.ndim == 0:
            r_update = np.array([r.item()])
        else:
            r_update = r
        self.ret_rms.update(r_update)
        norm_r = r / (self.ret_rms.std + self.epsilon)
        return np.clip(norm_r, -self.clip, self.clip)


#####################
#   工具函数         #
#####################
def get_available_port(start_port, end_port):
    """
    在指定范围内查找可用端口。
    """
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError(f"在范围 [{start_port}, {end_port}] 内未找到可用端口。")


def kill_sumo_processes():
    """
    杀死所有残留的 SUMO 进程。
    """
    killed = False
    try:
        if os.name == 'nt':  # Windows
            result1 = os.system("taskkill /f /im sumo.exe >nul 2>&1")
            result2 = os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
            killed = (result1 == 0 or result2 == 0)
        else:  # Linux/macOS
            result1 = os.system("pkill -f sumo > /dev/null 2>&1")
            result2 = os.system("pkill -f sumo-gui > /dev/null 2>&1")
            killed = (result1 == 0 or result2 == 0)
    except Exception as e:
        logging.warning(f"终止 SUMO 进程时出错: {e}")
    time.sleep(0.1)


def linear_decay(start_val, end_val, total_steps, current_step):
    """
    线性衰减计算。
    """
    if current_step >= total_steps:
        return end_val
    fraction = min(1.0, current_step / total_steps)
    return start_val + (end_val - start_val) * fraction


def calculate_rolling_average(data, window):
    """
    计算数据的滚动平均值。
    """
    if len(data) < window:
        return np.array([])
    data_np = np.array(data, dtype=float)
    data_np[~np.isfinite(data_np)] = 0
    return np.convolve(data_np, np.ones(window) / window, mode='valid')


#####################
#   SUMO 环境封装    #
#####################
class SumoEnv:
    """
    SUMO 仿真环境的封装类。
    """
    def __init__(self, config: Config):
        self.config = config
        self.sumo_process = None
        self.traci_port = None
        self.last_speed = 0.0
        self.last_raw_state = np.zeros(config.state_dim)
        self.last_norm_state = np.zeros(config.state_dim)
        self.last_lane_idx = 0
        self.num_lanes = config.num_train_lanes

        # 归一化
        self.obs_normalizer = RunningMeanStd(shape=(config.state_dim,), alpha=config.norm_update_rate) if config.normalize_observations else None
        self.reward_normalizer = RewardNormalizer(gamma=config.gamma, alpha=config.norm_update_rate, clip=config.reward_norm_clip) if config.normalize_rewards else None

        # 指标
        self.reset_metrics()

    def reset_metrics(self):
        """
        重置回合指标。
        """
        self.change_lane_count = 0
        self.collision_occurred = False
        self.current_step = 0
        self.last_action = 0

    def _start_sumo(self):
        """
        启动 SUMO 实例并连接 TraCI。
        """
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.config.port_range[0], self.config.port_range[1])
        except IOError as e:
            logging.error(f"无法找到可用端口: {e}")
            sys.exit(1)

        sumo_cmd = [
            self.config.sumo_binary, "-c", self.config.config_path,
            "--remote-port", str(self.traci_port),
            "--step-length", str(self.config.step_length),
            "--collision.check-junctions", "true",
            "--collision.action", "warn",
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--seed", str(np.random.randint(0, 10000))
        ]
        if self.config.sumo_binary == "sumo-gui":
            sumo_cmd.extend(["--start", "--quit-on-end"])

        try:
            stdout_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
            stderr_target = subprocess.DEVNULL if self.config.sumo_binary == "sumo" else None
            self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
        except FileNotFoundError:
            logging.error(f"SUMO 可执行文件 '{self.config.sumo_binary}' 未找到。")
            sys.exit(1)
        except Exception as e:
            logging.error(f"无法启动 SUMO 进程: {e}")
            sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5)
                traci.init(self.traci_port)
                return
            except traci.exceptions.TraCIException:
                if attempt == connection_attempts - 1:
                    logging.error("达到最大 TraCI 连接尝试次数。")
                    self._close()
                    raise ConnectionError(f"无法连接到 SUMO (端口: {self.traci_port})。")
            except Exception as e:
                logging.error(f"连接 TraCI 时发生意外错误: {e}")
                self._close()
                raise ConnectionError(f"连接到 SUMO 时发生未知错误 (端口: {self.traci_port})。")

    def _add_ego_vehicle(self):
        """
        将 Ego 车辆添加到仿真中。
        """
        ego_route_id = "route_E0"
        if ego_route_id not in traci.route.getIDList():
            try:
                traci.route.add(ego_route_id, ["E0"])
            except traci.exceptions.TraCIException as e:
                raise RuntimeError(f"添加路径 '{ego_route_id}' 失败: {e}")

        if self.config.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.config.ego_type_id)
                traci.vehicletype.setParameter(self.config.ego_type_id, "color", "1,0,0")  # Red
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcStrategic", "1.0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcSpeedGain", "2.0")
                traci.vehicletype.setParameter(self.config.ego_type_id, "lcCooperative", "0.5")
                traci.vehicletype.setParameter(self.config.ego_type_id, "jmIgnoreFoeProb", "0.1")
                traci.vehicletype.setParameter(self.config.ego_type_id, "carFollowModel", "IDM")
                traci.vehicletype.setParameter(self.config.ego_type_id, "minGap", "2.5")
            except traci.exceptions.TraCIException as e:
                logging.warning(f"设置 Ego 类型 '{self.config.ego_type_id}' 参数失败: {e}")

        if self.config.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.config.ego_vehicle_id)
                time.sleep(0.1)
            except traci.exceptions.TraCIException as e:
                logging.warning(f"移除残留 Ego 失败: {e}")

        try:
            start_lane = random.choice(range(self.num_lanes))
            traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                              typeID=self.config.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max")

            wait_steps = int(2.0 / self.config.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                    ego_appeared = True
                    break
            if not ego_appeared:
                logging.warning(f"在车道 {start_lane} 上添加 Ego 失败，尝试随机车道。")
                traci.vehicle.add(vehID=self.config.ego_vehicle_id, routeID=ego_route_id,
                                  typeID=self.config.ego_type_id, depart="now",
                                  departLane="random", departSpeed="max")
                for _ in range(wait_steps):
                    traci.simulationStep()
                    if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                        ego_appeared = True
                        break
                if not ego_appeared:
                    raise RuntimeError(f"Ego 车辆在 {wait_steps * 2} 步内未出现。")

        except traci.exceptions.TraCIException as e:
            logging.error(f"添加 Ego 车辆 '{self.config.ego_vehicle_id}' 失败: {e}")
            raise RuntimeError("添加 Ego 车辆失败。")
        except RuntimeError as e:
            logging.error(f"运行时错误: {e}")
            self._close()
            raise

    def reset(self) -> np.ndarray:
        """
        为新回合重置环境，返回归一化状态。
        """
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.reset_metrics()

        self.last_speed = 0.0
        self.last_raw_state = np.zeros(self.config.state_dim)
        self.last_norm_state = np.zeros(self.config.state_dim)
        self.last_lane_idx = 0
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
                logging.warning(f"reset 中的初始状态获取期间发生 TraCI 异常: {e}")
                norm_state = np.zeros(self.config.state_dim)
            except IndexError:
                logging.warning("访问 reset 中的初始状态时发生 IndexError。")
                norm_state = np.zeros(self.config.state_dim)
        else:
            logging.warning("在 reset 中的 add/wait 后未立即找到 Ego 车辆。")
            try:
                traci.simulationStep()
            except traci.exceptions.TraCIException:
                pass
            if self.config.ego_vehicle_id in traci.vehicle.getIDList():
                return self.reset()
            else:
                logging.error("Ego 车辆在 reset 后仍然不存在。")
                norm_state = np.zeros(self.config.state_dim)

        return norm_state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        使用运行均值/标准差归一化状态。
        """
        if self.obs_normalizer:
            self.obs_normalizer.update(state.reshape(1, -1))
            norm_state = (state - self.obs_normalizer.mean) / self.obs_normalizer.std
            norm_state = np.clip(norm_state, -self.config.obs_norm_clip, self.config.obs_norm_clip)
            if np.any(np.isnan(norm_state)) or np.any(np.isinf(norm_state)):
                return self.last_norm_state.copy()
            return norm_state.astype(np.float32)
        else:
            return state

    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> typing.Dict[str, typing.Tuple[float, float]]:
        """
        获取周围车辆的距离和相对速度。
        """
        max_dist = self.config.max_distance
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        try:
            veh_ids = traci.vehicle.getIDList()
            if ego_id not in veh_ids:
                return infos
        except traci.exceptions.TraCIException:
            return infos

        try:
            ego_road = traci.vehicle.getRoadID(ego_id)
            if not ego_road or not ego_road.startswith("E"):
                return infos
        except traci.exceptions.TraCIException:
            return infos

        for veh_id in veh_ids:
            if veh_id == ego_id:
                continue
            try:
                if traci.vehicle.getRoadID(veh_id) != ego_road:
                    continue

                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)

                if not (0 <= veh_lane < self.num_lanes):
                    continue

                dx = veh_pos[0] - ego_pos[0]
                distance = abs(dx)

                if distance >= max_dist:
                    continue
                rel_speed = ego_speed - veh_speed

                if veh_lane == ego_lane:
                    if dx > 0 and distance < infos['front'][0]:
                        infos['front'] = (distance, rel_speed)
                elif veh_lane == ego_lane - 1:
                    if dx > 0 and distance < infos['left_front'][0]:
                        infos['left_front'] = (distance, rel_speed)
                    elif dx <= 0 and distance < infos['left_back'][0]:
                        infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1:
                    if dx > 0 and distance < infos['right_front'][0]:
                        infos['right_front'] = (distance, rel_speed)
                    elif dx <= 0 and distance < infos['right_back'][0]:
                        infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue
        return infos

    def _get_raw_state(self) -> np.ndarray:
        """
        获取当前环境状态 (归一化之前的原始值)。
        """
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        ego_id = self.config.ego_vehicle_id

        try:
            if ego_id not in traci.vehicle.getIDList():
                return self.last_raw_state
        except traci.exceptions.TraCIException:
            return self.last_raw_state

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road or not current_road.startswith("E"):
                return self.last_raw_state

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)

            if not (0 <= ego_lane < self.num_lanes):
                time.sleep(0.05)
                if ego_id in traci.vehicle.getIDList():
                    try:
                        ego_lane = traci.vehicle.getLaneIndex(ego_id)
                    except traci.exceptions.TraCIException:
                        ego_lane = self.last_lane_idx
                else:
                    ego_lane = self.last_lane_idx
                if not (0 <= ego_lane < self.num_lanes):
                    ego_lane = self.last_lane_idx
                ego_lane = np.clip(ego_lane, 0, self.num_lanes-1)


            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (self.num_lanes - 1) else False

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
                logging.warning("检测到 NaN 或 Inf，使用最后有效的原始状态。")
                return self.last_raw_state.copy()

            self.last_raw_state = state.copy()

        except traci.exceptions.TraCIException as e:
            is_not_known = "Vehicle '" + ego_id + "' is not known" in str(e)
            is_teleporting = "teleporting" in str(e).lower()
            if not is_not_known and not is_teleporting:
                logging.warning(f"获取 {ego_id} 的原始状态时发生 TraCI 错误: {e}。返回最后已知的原始状态。")
            return self.last_raw_state.copy()
        except Exception as e:
            logging.warning(f"获取 {ego_id} 的原始状态时发生未知错误: {e}。");
            traceback.print_exc()
            return self.last_raw_state.copy()

        return state

    def step(self, action: int) -> typing.Tuple[np.ndarray, float, bool, int]:
        """
        执行一个动作，返回 (next_normalized_state, normalized_reward, done, next_lane_index)。
        """
        done = False
        raw_reward = 0.0
        next_lane_index = self.last_lane_idx
        ego_id = self.config.ego_vehicle_id
        self.last_action = action

        try:
            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True
                done = True
                coll_reward_raw = self.config.reward_collision
                norm_coll_reward = self.reward_normalizer.normalize(
                    np.array([coll_reward_raw]))[0] if self.reward_normalizer else coll_reward_raw
                return self.last_norm_state.copy(), norm_coll_reward, True, self.last_lane_idx
        except traci.exceptions.TraCIException:
            self.collision_occurred = True
            done = True
            coll_reward_raw = self.config.reward_collision
            norm_coll_reward = self.reward_normalizer.normalize(
                np.array([coll_reward_raw]))[0] if self.reward_normalizer else coll_reward_raw
            return self.last_norm_state.copy(), norm_coll_reward, True, self.last_lane_idx

        try:
            previous_lane_idx = self.last_lane_idx
            current_lane = previous_lane_idx

            lane_change_requested = False
            if action == 1:
                if current_lane > 0 and traci.vehicle.couldChangeLane(ego_id, -1):
                    traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
                    lane_change_requested = True
            elif action == 2:
                if current_lane < (self.num_lanes - 1) and traci.vehicle.couldChangeLane(ego_id, 1):
                    traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
                    lane_change_requested = True

            traci.simulationStep()
            self.current_step += 1

            if ego_id not in traci.vehicle.getIDList():
                self.collision_occurred = True
                done = True
                raw_reward = self.config.reward_collision
                next_norm_state = self.last_norm_state.copy()
                next_lane_index = self.last_lane_idx
                normalized_reward = self.reward_normalizer.normalize(
                    np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward
                return next_norm_state, normalized_reward, done, next_lane_index

            collisions = traci.simulation.getCollisions()
            ego_collided = False
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True
                    ego_collided = True
                    done = True
                    raw_reward = self.config.reward_collision
                    break

            next_raw_state = self._get_raw_state()
            next_norm_state = self._normalize_state(next_raw_state)
            next_lane_index = np.clip(int(round(next_raw_state[1])), 0, self.num_lanes - 1)

            if not self.collision_occurred:
                current_speed_after_step = next_raw_state[0]
                front_dist_after_step = next_raw_state[2]
                left_front_dist = next_raw_state[4]
                left_back_dist = next_raw_state[6]
                right_front_dist = next_raw_state[7]
                right_back_dist = next_raw_state[9]
                can_change_left_after_step = next_raw_state[10] > 0.5
                can_change_right_after_step = next_raw_state[11] > 0.5

                actual_lane_change = (previous_lane_idx != next_lane_index)
                if actual_lane_change:
                    self.change_lane_count += 1

                raw_reward = self._calculate_reward(
                    action_taken=action,
                    current_speed=current_speed_after_step,
                    current_lane=next_lane_index,
                    front_dist=front_dist_after_step,
                    previous_lane=previous_lane_idx,
                    can_change_left=can_change_left_after_step,
                    can_change_right=can_change_right_after_step,
                    left_front_dist=left_front_dist,
                    left_back_dist=left_back_dist,
                    right_front_dist=right_front_dist,
                    right_back_dist=right_back_dist,
                    lane_change_requested=lane_change_requested)

            if traci.simulation.getTime() >= 3600: # Max simulation time
                done = True
            if self.current_step >= self.config.max_steps:
                done = True

        except traci.exceptions.TraCIException as e:
            is_known_error = "Vehicle '" + ego_id + "' is not known" in str(
                e) or "teleporting" in str(e).lower()
            if not is_known_error:
                logging.error(f"在步骤 {self.current_step} 期间发生 TraCI 异常: {e}")
            if is_known_error or (ego_id not in traci.vehicle.getIDList() if not is_known_error else False): # check if ego exists only if not known_error
                self.collision_occurred = True
            raw_reward = self.config.reward_collision
            done = True
            next_norm_state = self.last_norm_state.copy()
            next_lane_index = self.last_lane_idx
        except Exception as e:
            logging.error(f"在步骤 {self.current_step} 期间发生未知异常: {e}")
            traceback.print_exc()
            done = True
            raw_reward = self.config.reward_collision
            self.collision_occurred = True
            next_norm_state = self.last_norm_state.copy()
            next_lane_index = self.last_lane_idx

        if not (np.any(np.isnan(next_raw_state)) or np.any(np.isinf(next_raw_state))):
            self.last_speed = next_raw_state[0]
        self.last_lane_idx = next_lane_index
        if not (np.any(np.isnan(next_norm_state)) or np.any(np.isinf(next_norm_state))):
            self.last_norm_state = next_norm_state.copy()

        normalized_reward = self.reward_normalizer.normalize(
            np.array([raw_reward]))[0] if self.reward_normalizer else raw_reward
        if not np.isfinite(normalized_reward):
            normalized_reward = 0.0

        return self.last_norm_state.copy(), normalized_reward, done, next_lane_index

    def _calculate_reward(self, action_taken: int, current_speed: float, current_lane: int,
                           front_dist: float, previous_lane: int,
                           can_change_left: bool, can_change_right: bool,
                           left_front_dist: float, left_back_dist: float,
                           right_front_dist: float, right_back_dist: float,
                           lane_change_requested: bool) -> float:
        """
        计算奖励。
        """
        if self.collision_occurred: # This should already be handled by raw_reward in step() before calling this
            return 0.0 # Should not be reached if collision_occurred is true, but as safeguard.

        try:
            current_lane = np.clip(current_lane, 0, self.num_lanes - 1)
            previous_lane = np.clip(previous_lane, 0, self.num_lanes - 1)

            lane_max_speed = self.config.lane_max_speeds[current_lane]
            target_speed = lane_max_speed * self.config.target_speed_factor

            speed_diff = abs(current_speed - target_speed)
            norm_target_speed = max(lane_max_speed * 0.5, 1.0)
            speed_reward = np.exp(- (speed_diff / norm_target_speed) ** 2) * self.config.reward_high_speed_scale

            low_speed_penalty = 0.0
            low_speed_threshold = target_speed * 0.6
            if current_speed < low_speed_threshold and target_speed > 1.0:
                low_speed_penalty = (
                                            current_speed / max(low_speed_threshold, 1.0) - 1.0) * self.config.reward_low_speed_penalty_scale

            lane_change_actual = (current_lane != previous_lane)
            lane_change_reward_penalty = 0.0
            if lane_change_actual:
                lane_change_reward_penalty += self.config.reward_lane_change_penalty
                if self.config.lane_max_speeds[current_lane] > self.config.lane_max_speeds[previous_lane]:
                    lane_change_reward_penalty += self.config.reward_faster_lane_bonus

            staying_slow_penalty = 0.0
            if not lane_change_actual and action_taken == 0:
                left_lane_idx = current_lane - 1
                if left_lane_idx >= 0 and self.config.lane_max_speeds[left_lane_idx] > lane_max_speed:
                    if can_change_left and left_front_dist > self.config.min_safe_change_dist and left_back_dist > self.config.min_safe_change_dist * 0.6:
                        staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale
                elif staying_slow_penalty == 0.0: # Only check right if left wasn't a better option
                    right_lane_idx = current_lane + 1
                    if right_lane_idx < self.num_lanes and self.config.lane_max_speeds[right_lane_idx] > lane_max_speed:
                        if can_change_right and right_front_dist > self.config.min_safe_change_dist and right_back_dist > self.config.min_safe_change_dist * 0.6:
                            staying_slow_penalty = -self.config.reward_staying_slow_penalty_scale

            safety_dist_penalty = 0.0
            min_safe_dist = self.config.min_buffer_dist_reward + current_speed * self.config.time_gap_reward
            if front_dist < self.config.max_distance and front_dist < min_safe_dist: # Check if front vehicle is within detection range
                safety_dist_penalty = (
                                               front_dist / max(min_safe_dist, 1e-6) - 1.0) * self.config.safe_distance_penalty_scale

            comfort_penalty = 0.0
            acceleration = (current_speed - self.last_speed) / self.config.step_length
            harsh_braking_threshold = -3.0
            if acceleration < harsh_braking_threshold:
                comfort_penalty = (acceleration - harsh_braking_threshold) * self.config.reward_comfort_penalty_scale

            time_alive = self.config.time_alive_reward

            total_reward = (speed_reward +
                            low_speed_penalty +
                            lane_change_reward_penalty +
                            staying_slow_penalty +
                            safety_dist_penalty +
                            comfort_penalty +
                            time_alive)

            if not np.isfinite(total_reward):
                # logging.warning(f"Calculated raw reward is non-finite. Components: speed_r={speed_reward}, low_s_p={low_speed_penalty}, lc_p={lane_change_reward_penalty}, stay_s_p={staying_slow_penalty}, safe_p={safety_dist_penalty}, comf_p={comfort_penalty}")
                total_reward = 0.0

            return total_reward

        except IndexError as e_idx:
            logging.warning(f"计算奖励时发生 IndexError (可能是无效的车道索引 {current_lane} 或 {previous_lane}): {e_idx}。返回 0。")
            traceback.print_exc()
            return 0.0
        except Exception as e:
            logging.warning(f"计算奖励时出错: {e}。返回 0。");
            traceback.print_exc()
            return 0.0

    def _close(self):
        """
        关闭 SUMO 实例和 TraCI 连接。
        """
        if self.sumo_process:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError: # Connection might be already closed
                pass
            except Exception: # Other potential errors on close
                pass
            finally:
                try:
                    if self.sumo_process.poll() is None: # Check if process is still running
                        self.sumo_process.terminate()
                        self.sumo_process.wait(timeout=2) # Wait for termination
                except subprocess.TimeoutExpired:
                    self.sumo_process.kill() # Force kill if terminate fails
                    self.sumo_process.wait(timeout=1)
                except Exception as e: # Catch other errors during termination
                    logging.warning(f"SUMO 终止期间出错: {e}")
                self.sumo_process = None
                self.traci_port = None
                time.sleep(0.1) # Brief pause after closing
        else:
            self.traci_port = None # Ensure port is cleared if no process


#####################
#   DQN 组件         #
#####################
NStepTransition = namedtuple('NStepTransition',
                             ('state', 'action', 'reward', 'next_state', 'done', 'next_lane_index'))
Experience = namedtuple('Experience',
                          ('state', 'action', 'n_step_reward', 'next_state', 'done', 'n', 'next_lane_index'))


class SumTree:
    """
    SumTree 用于 Prioritized Experience Replay。
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        if dataIdx < 0 or dataIdx >= self.capacity: # Robustness check
            logging.warning(f"从 SumTree 检索到无效的 dataIdx {dataIdx} (tree idx {idx}, s={s}, total={self.total()})。返回虚拟数据。")
            dummy_state = np.zeros(Config.state_dim) # Use state_dim from Config
            return (idx, 0.0, Experience(dummy_state, 0, 0, dummy_state, True, 1, 0))

        data = self.data[dataIdx]
        if not isinstance(data, Experience): # Robustness check for corrupted data
            logging.warning(
                f"在 SumTree 的 dataIdx {dataIdx} (tree idx {idx}) 处发现损坏的数据。 data: {data}。返回虚拟数据。")
            dummy_state = np.zeros(Config.state_dim)
            return (idx, self.tree[idx], Experience(dummy_state, 0, 0, dummy_state, True, 1, 0))
        return (idx, self.tree[idx], data)

    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer (PER + N-Step)。
    """
    def __init__(self, capacity: int, alpha: float, config: Config):
        self.config = config
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, experience: Experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int, beta: float) -> typing.Tuple[typing.List[Experience], np.ndarray, np.ndarray]:
        experiences = []
        indices = np.empty((batch_size,), dtype=np.int32)
        is_weights = np.empty((batch_size,), dtype=np.float32)
        segment = self.tree.total() / batch_size
        if self.tree.total() <= 0 or batch_size <= 0 or segment <= 0:
            logging.warning("SumTree 总和为零或 batch_size 为零，无法采样。返回空列表。")
            return [], np.array([]), np.array([])

        for i in range(batch_size):
            attempt = 0
            successful_sample = False
            while attempt < 10: # Retry sampling if invalid data encountered
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, min(b, self.tree.total()))
                s = max(0.0, min(s, self.tree.total() - 1e-9)) if self.tree.total() > 0 else 0.0 # Clamp s

                try:
                    (idx, p, data) = self.tree.get(s)

                    if not isinstance(data, Experience) or not isinstance(data.state,
                                                                       np.ndarray) or data.state.shape != (
                            self.config.state_dim,):
                        logging.warning(
                            f"在索引 {idx} 处采样到无效数据 (类型: {type(data)}), tree 总和 {self.tree.total()}, segment [{a:.4f},{b:.4f}], s {s:.4f}。重试采样 ({attempt + 1}/10)...")
                        attempt += 1; time.sleep(0.001)
                        segment = self.tree.total() / batch_size if batch_size > 0 and self.tree.total() > 0 else 0
                        if segment <=0: break # Stop retrying if segment becomes invalid
                        continue

                    experiences.append(data)
                    indices[i] = idx
                    prob = p / (self.tree.total() + 1e-9) # Add epsilon for stability
                    prob = max(prob, 1e-9) # Ensure probability is positive

                    is_weights[i] = np.power(self.tree.n_entries * prob, -beta)
                    successful_sample = True
                    break # Sample successful
                except Exception as e:
                    logging.error(f"为 s={s} (range [{a},{b}]), idx={idx}, tree total={self.tree.total()} 执行 SumTree.get 或处理时出错: {e}")
                    traceback.print_exc()
                    attempt += 1; time.sleep(0.001)
                    segment = self.tree.total() / batch_size if batch_size > 0 and self.tree.total() > 0 else 0
                    if segment <= 0: break # Stop retrying

            if not successful_sample:
                # logging.warning(f"Attempted {attempt} times, failed to sample valid Experience data from SumTree. Tree total: {self.tree.total()}, n_entries: {self.tree.n_entries}")
                return [], np.array([]), np.array([]) # Return empty if any sample fails after retries

        max_weight = is_weights.max()
        is_weights /= (max_weight + 1e-8) # Normalize weights, add epsilon

        if np.any(~np.isfinite(is_weights)):
            logging.warning(f"IS 权重包含 NaN 或 Inf。 Weights: {is_weights}")
            is_weights[~np.isfinite(is_weights)] = 1.0 # Replace with neutral weight

        return experiences, is_weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + self.config.per_epsilon
        for idx, priority in zip(indices, priorities):
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                logging.warning(f"提供给 update_priorities 的索引 {idx} 无效 (capacity={self.tree.capacity})。跳过。")
                continue
            if not np.isfinite(priority):
                # logging.warning(f"Priority for index {idx} is non-finite ({priority}). Clipping.") # Reduce log spam
                priority = self.max_priority
            elif priority <= 0: # Ensure priority is positive
                # logging.warning(f"Priority for index {idx} is <= 0 ({priority}). Using epsilon.") # Reduce log spam
                priority = self.config.per_epsilon

            self.max_priority = max(self.max_priority, priority)
            try:
                self.tree.update(idx, priority ** self.alpha)
            except Exception as e:
                logging.error(f"更新 SumTree 索引 {idx} 优先级 {priority} 时出错: {e}")
                continue # Skip problematic update

    def __len__(self) -> int:
        return self.tree.n_entries


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer with Factorized Gaussian Noise.
    """
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
        x = torch.randn(size, device=self.weight_mu.device) # Ensure noise is on the same device
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise() # Reset noise at each forward pass during training
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon.to(self.weight_sigma.device))
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon.to(self.bias_sigma.device))
        else: # Use mean weights/biases during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    """
    Q-Network (Modified for C51 and Noisy Nets).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, config: Config):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = config.num_atoms
        self.use_noisy = config.use_noisy_nets
        self.sigma_init = config.noisy_sigma_init

        linear = lambda in_f, out_f: NoisyLinear(in_f, out_f,
                                                 self.sigma_init) if self.use_noisy else nn.Linear(in_f, out_f)

        self.feature_layer = nn.Sequential(
            linear(state_dim, hidden_size), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.num_atoms)
        )
        self.advantage_stream = nn.Sequential(
            linear(hidden_size, hidden_size), nn.ReLU(),
            linear(hidden_size, self.action_dim * self.num_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.feature_layer(x)

        values_logits = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantages_logits = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)

        q_logits = values_logits + advantages_logits - advantages_logits.mean(dim=1, keepdim=True)

        q_probs = F.softmax(q_logits, dim=2) # Softmax over atoms dimension

        q_probs = q_probs.clamp(min=1e-8) # Clamp for numerical stability

        return q_probs

    def reset_noise(self):
        """
        Resets noise in all NoisyLinear layers.
        """
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class DQNAgent:
    """
    DQN 代理类。
    """
    def __init__(self, config: Config):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1) if self.num_atoms > 1 else 0
        self.num_lanes = config.num_train_lanes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")
        self.support = self.support.to(self.device)

        self.policy_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.initial_learning_rate, eps=1e-5)

        # V2: Initialize Learning Rate Scheduler
        if config.lr_decay_total_updates > 0: # Only if decay is configured
            self.scheduler = LinearLR(self.optimizer,
                                      start_factor=1.0,
                                      end_factor=config.lr_end_factor,
                                      total_iters=config.lr_decay_total_updates)
            logging.info(f"学习率调度器已启用: LinearLR, total_iters={config.lr_decay_total_updates}, end_factor={config.lr_end_factor}")
        else:
            self.scheduler = None
            logging.info("学习率调度器未启用 (lr_decay_total_updates <= 0)。")


        if config.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size, config.per_alpha, config)
        else:
            logging.warning("PER 已禁用，使用标准回放缓冲区。")
            # This part would need a standard replay buffer implementation if PER is False
            self.replay_buffer = deque([], maxlen=config.replay_buffer_size)
            raise NotImplementedError("Standard Replay Buffer push/sample logic not fully implemented here, designed for PER.")

        self.train_step_count = 0
        self.loss_history = []

    def get_action(self, normalized_state: np.ndarray, current_lane_idx: int) -> int:
        """
        选择动作。
        """
        self.policy_net.eval() # Set to eval for action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(state_tensor)
            expected_q_values = (action_probs * self.support.view(1, 1, -1)).sum(dim=2)

            q_values_masked = expected_q_values.clone()
            current_lane_idx = np.clip(current_lane_idx, 0, self.num_lanes - 1)
            if current_lane_idx == 0: # Cannot change left from lane 0
                q_values_masked[0, 1] = -float('inf')
            if current_lane_idx >= self.num_lanes - 1: # Cannot change right from the rightmost lane
                q_values_masked[0, 2] = -float('inf')

            action = q_values_masked.argmax().item()

        self.policy_net.train() # Set back to train mode
        return action

    def update(self, global_step: int) -> typing.Optional[float]:
        """
        更新 Q-network。
        """
        if len(self.replay_buffer) < max(self.config.learning_starts, self.config.batch_size):
            return None

        current_beta = linear_decay(self.config.per_beta_start, self.config.per_beta_end,
                                    self.config.per_beta_annealing_steps, global_step)

        try:
            if self.config.use_per:
                experiences, is_weights, indices = self.replay_buffer.sample(self.config.batch_size, current_beta)
                if not experiences: # Handle empty sample case from PER
                    # logging.warning("Failed to sample from PER. Skipping update.") # Reduce log spam
                    return None
            else:
                raise NotImplementedError("Standard Replay Buffer sampling logic needed if PER is off.")
        except RuntimeError as e:
            logging.error(f"从回放缓冲区采样时出错: {e}。跳过更新。");
            return None
        except Exception as e_sample:
            logging.error(f"从回放缓冲区采样时发生意外错误: {e_sample}。跳过更新。");
            traceback.print_exc();
            return None

        try:
            batch = Experience(*zip(*experiences))

            # Validate states and other batch components before tensor conversion
            valid_idx = [i for i, s in enumerate(batch.state) if
                         s is not None and isinstance(s, np.ndarray) and s.shape == (self.state_dim,)]
            if len(valid_idx) != self.config.batch_size: # If not all samples are valid
                # logging.warning(f"Sampled batch contains invalid states ({self.config.batch_size - len(valid_idx)} / {self.config.batch_size}). Skipping update.") # Reduce log spam
                return None # Or handle partial batch if implemented

            states_np = np.array([batch.state[i] for i in valid_idx], dtype=np.float32)
            actions_np = np.array([batch.action[i] for i in valid_idx], dtype=np.int64)
            n_step_rewards_np = np.array([batch.n_step_reward[i] for i in valid_idx], dtype=np.float32)
            next_states_np = np.array([batch.next_state[i] for i in valid_idx], dtype=np.float32)
            dones_np = np.array([batch.done[i] for i in valid_idx], dtype=bool)
            n_np = np.array([batch.n[i] for i in valid_idx], dtype=np.float32)
            next_lane_indices_np = np.array([batch.next_lane_index[i] for i in valid_idx], dtype=np.int64)


            # Ensure rewards are finite before tensor conversion
            if not np.all(np.isfinite(n_step_rewards_np)):
                # logging.warning("Batch rewards contain NaN/Inf. Clipping/replacing before update.") # Reduce log spam
                n_step_rewards_np = np.nan_to_num(n_step_rewards_np, nan=0.0,
                                                 posinf=self.config.reward_norm_clip,
                                                 neginf=-self.config.reward_norm_clip)

            states = torch.from_numpy(states_np).to(self.device)
            actions = torch.from_numpy(actions_np).to(self.device)
            rewards = torch.from_numpy(n_step_rewards_np).to(self.device)
            next_states = torch.from_numpy(next_states_np).to(self.device)
            dones = torch.from_numpy(dones_np).to(self.device)
            is_weights_tensor = torch.from_numpy(np.array(is_weights, dtype=np.float32)).to(self.device)
            gammas = torch.pow(self.config.gamma, torch.from_numpy(n_np).to(self.device))
            next_lane_indices = torch.from_numpy(next_lane_indices_np).to(self.device)

        except Exception as e_tensor:
            logging.error(f"将批次转换为张量时出错: {e_tensor}");
            traceback.print_exc();
            return None

        with torch.no_grad():
            next_dist_target = self.target_net(next_states)

            if self.config.use_double_dqn:
                self.policy_net.eval() # Use policy_net (in eval mode) to select best actions
                next_dist_policy = self.policy_net(next_states)
                self.policy_net.train() # Switch back to train mode

                next_expected_q_policy = (next_dist_policy * self.support.view(1, 1, -1)).sum(dim=2)
                q_values_masked = next_expected_q_policy.clone()
                q_values_masked[next_lane_indices == 0, 1] = -float('inf')
                q_values_masked[next_lane_indices >= self.num_lanes - 1, 2] = -float('inf')
                best_next_actions = q_values_masked.argmax(dim=1)
            else: # Standard DQN action selection from target net
                next_expected_q_target = (next_dist_target * self.support.view(1, 1, -1)).sum(dim=2)
                q_values_masked = next_expected_q_target.clone()
                q_values_masked[next_lane_indices == 0, 1] = -float('inf')
                q_values_masked[next_lane_indices >= self.num_lanes - 1, 2] = -float('inf')
                best_next_actions = q_values_masked.argmax(dim=1)

            best_next_dist = next_dist_target[torch.arange(states.size(0)), best_next_actions, :]

            Tz = rewards.unsqueeze(1) + gammas.unsqueeze(1) * self.support.unsqueeze(0) * (
                ~dones).unsqueeze(1).float()
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            if self.delta_z > 0:
                b = (Tz - self.v_min) / self.delta_z
                lower_idx = b.floor().long()
                upper_idx = b.ceil().long()
                lower_idx.clamp_(min=0, max=self.num_atoms - 1)
                upper_idx.clamp_(min=0, max=self.num_atoms - 1)

                lower_weight = (upper_idx.float() - b).clamp(0, 1)
                upper_weight = (b - lower_idx.float()).clamp(0, 1)
            else: # Handle single atom case (num_atoms = 1)
                lower_idx = torch.zeros_like(Tz).long()
                upper_idx = torch.zeros_like(Tz).long()
                lower_weight = torch.ones_like(Tz)
                upper_weight = torch.zeros_like(Tz)

            target_dist_projected = torch.zeros_like(best_next_dist)

            batch_indices = torch.arange(states.size(0), device=self.device)

            # Flatten for index_put_
            flat_lower_idx = lower_idx.flatten()
            flat_upper_idx = upper_idx.flatten()
            flat_best_next_dist = best_next_dist.flatten()
            flat_lower_weight = lower_weight.flatten()
            flat_upper_weight = upper_weight.flatten()
            flat_batch_indices = batch_indices.unsqueeze(1).expand(-1, self.num_atoms).flatten()

            target_dist_projected.index_put_(
                (flat_batch_indices, flat_lower_idx), flat_best_next_dist * flat_lower_weight, accumulate=True)
            target_dist_projected.index_put_(
                (flat_batch_indices, flat_upper_idx), flat_best_next_dist * flat_upper_weight, accumulate=True)

        current_dist_all_actions = self.policy_net(states)
        current_dist = current_dist_all_actions[torch.arange(states.size(0)), actions, :]

        elementwise_loss = -(target_dist_projected.detach() * (current_dist + 1e-8).log()).sum(dim=1)

        td_errors = elementwise_loss.detach()

        if self.config.use_per and indices is not None:
             td_errors_np = td_errors.cpu().numpy()
             if np.all(np.isfinite(td_errors_np)): # Ensure TD errors are finite
                 self.replay_buffer.update_priorities(indices, td_errors_np)
             else:
                 logging.warning("TD 错误包含 NaN/Inf。跳过优先级更新。")

        loss = (elementwise_loss * is_weights_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # V2: Step the LR scheduler if it's enabled and after optimizer.step()
        if self.scheduler and self.train_step_count >= self.config.learning_starts : # Only step scheduler after learning starts
            self.scheduler.step()

        self.train_step_count += 1
        self.loss_history.append(loss.item())

        if self.train_step_count % self.config.target_update_freq == 0:
            logging.info(f"--- 更新目标网络 @ 步骤 {self.train_step_count} ---")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


#####################
#   主训练循环       #
#####################
def main():
    """
    主训练函数。
    """
    config = Config()

    # --- 日志配置 ---
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"dqn_training_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filepath, encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])# --- 基本检查和设置 ---
    required_files = [config.config_path, "a.net.xml", "a.rou.xml"]
    abort_training = False
    for f in required_files:
        if not os.path.exists(f):
            logging.error(f"未找到所需的 SUMO 文件: {f}")
            abort_training = True
    if abort_training:
        sys.exit(1)

    # --- Env 和 Agent 初始化 ---
    env = SumoEnv(config)
    agent = DQNAgent(config)

    # --- 日志和保存设置 ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = "dqn_v2" # Updated experiment name
    if config.use_per:
        exp_name += "_per"
    if config.use_n_step:
        exp_name += f"_n{config.n_step}"
    if config.use_distributional:
        exp_name += "_c51"
    if config.use_noisy_nets:
        exp_name += "_noisy"
    if config.use_double_dqn:
        exp_name += "_ddqn"
    results_dir = f"{exp_name}_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    logging.info(f"结果将保存在: {results_dir}")
    logging.info(
        f"配置: PER={config.use_per}, N-Step={config.n_step if config.use_n_step else 'Off'}, C51={config.use_distributional}, Noisy={config.use_noisy_nets}, DDQN={config.use_double_dqn}")
    logging.info(
        f"LR_Initial={config.initial_learning_rate}, LR_End_Factor={config.lr_end_factor if agent.scheduler else 'N/A'}, LR_Decay_Updates={config.lr_decay_total_updates if agent.scheduler else 'N/A'}")
    logging.info(
        f"TargetUpdate={config.target_update_freq}, StartLearn={config.learning_starts}, BatchSize={config.batch_size}")
    logging.info(
        f"奖励参数: FastLaneBonus={config.reward_faster_lane_bonus}, StaySlowPen={config.reward_staying_slow_penalty_scale}, LC_Penalty={config.reward_lane_change_penalty}, SpeedScale={config.reward_high_speed_scale}")
    logging.info(f"归一化: Obs={config.normalize_observations}, Reward={config.normalize_rewards}")
    logging.info(f"训练设备: {agent.device}")
    logging.info(f"训练回合数: {config.dqn_episodes} (建议增加以获得更好收敛)")


    try:
        config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
        config_save_path = os.path.join(results_dir, "config_used.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e:
        logging.warning(f"无法保存配置 JSON: {e}")

    # 指标列表
    all_rewards_sum_norm = []
    lane_change_counts = []
    collision_counts = []
    total_steps_per_episode = []
    avg_losses_per_episode = []
    beta_values = []
    learning_rates_history = [] # V2: Track learning rates
    best_avg_reward = -float('inf')
    global_step_count = 0
    n_step_buffer = deque(maxlen=config.n_step if config.use_n_step else 1)

    # --- DQN 训练循环 ---
    logging.info("\n" + "#" * 20 + f" 开始 DQN 训练 ({exp_name}) " + "#" * 20)
    try:
        for episode in tqdm(range(1, config.dqn_episodes + 1), desc="DQN 训练回合"):
            try:
                state_norm = env.reset()
            except (RuntimeError, ConnectionError) as e_reset:
                logging.error(f"回合 {episode} 重置环境失败: {e_reset}。跳过此回合。")
                time.sleep(2)
                continue

            if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                logging.error(f"回合 {episode} 的初始状态无效。尝试再次重置...")
                time.sleep(1)
                try:
                    state_norm = env.reset()
                except (RuntimeError, ConnectionError) as e_reset2:
                    logging.error(f"回合 {episode} 第二次重置环境失败: {e_reset2}。跳过此回合。")
                    time.sleep(2)
                    continue
                if np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                    logging.error("重置后初始状态仍然无效。中止训练。")
                    raise RuntimeError("无效的初始状态。")

            episode_reward_norm_sum = 0.0
            episode_loss_sum = 0.0
            episode_loss_count = 0
            done = False
            step_count = 0
            n_step_buffer.clear()

            while not done and step_count < config.max_steps:
                if env.last_raw_state is not None and not (
                        np.any(np.isnan(env.last_raw_state)) or np.any(np.isinf(env.last_raw_state))):
                    current_lane_idx = int(round(env.last_raw_state[1]))
                    current_lane_idx = np.clip(current_lane_idx, 0, env.num_lanes - 1)
                else:
                    current_lane_idx = env.last_lane_idx

                if state_norm is None or np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                    logging.warning(f"回合 {episode} 步骤 {step_count} 的状态无效。使用最后一个有效状态。")
                    state_norm = env.last_norm_state.copy()
                    if state_norm is None or np.any(np.isnan(state_norm)) or np.any(np.isinf(state_norm)):
                        logging.error("无法恢复有效状态。中止回合。")
                        done = True
                        break

                action = agent.get_action(state_norm, current_lane_idx)
                try:
                    next_state_norm, norm_reward, done, next_lane_idx = env.step(action)
                except traci.exceptions.FatalTraCIError as e_traci_fatal:
                    logging.error(f"回合 {episode} 步骤 {step_count} 发生致命 TraCI 错误: {e_traci_fatal}。中止回合。")
                    done = True
                    next_state_norm = state_norm # Use current state as next if error
                    norm_reward = env.reward_normalizer.normalize(np.array([config.reward_collision]))[0] if env.reward_normalizer else config.reward_collision
                    next_lane_idx = current_lane_idx
                    env.collision_occurred = True
                except Exception as e_step:
                    logging.error(f"回合 {episode} 步骤 {step_count} env.step 期间发生未知错误: {e_step}")
                    traceback.print_exc()
                    done = True
                    next_state_norm = state_norm
                    norm_reward = env.reward_normalizer.normalize(np.array([config.reward_collision]))[0] if env.reward_normalizer else config.reward_collision
                    next_lane_idx = current_lane_idx
                    env.collision_occurred = True

                if next_state_norm is None or np.any(np.isnan(next_state_norm)) or np.any(np.isinf(next_state_norm)):
                    next_state_norm = state_norm.copy()
                if not np.isfinite(norm_reward):
                    norm_reward = 0.0

                if config.use_n_step:
                    n_step_buffer.append(
                        NStepTransition(state_norm, action, norm_reward, next_state_norm, done, next_lane_idx))

                    if len(n_step_buffer) >= config.n_step or (done and len(n_step_buffer) > 0):
                        n_step_actual = len(n_step_buffer)
                        n_step_return_discounted = 0.0
                        for i in range(n_step_actual):
                            n_step_return_discounted += (config.gamma ** i) * n_step_buffer[i].reward

                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        s_t_plus_n = n_step_buffer[-1].next_state
                        done_n_step = n_step_buffer[-1].done
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index

                        exp = Experience(s_t, a_t, n_step_return_discounted, s_t_plus_n, done_n_step,
                                           n_step_actual, next_lane_idx_n_step)

                        try: # Robust push
                            if exp.state is not None and exp.next_state is not None and \
                               not (np.any(np.isnan(exp.state)) or np.any(np.isinf(exp.state))) and \
                               not (np.any(np.isnan(exp.next_state)) or np.any(np.isinf(exp.next_state))) and \
                               np.isfinite(exp.n_step_reward):
                                if config.use_per:
                                    agent.replay_buffer.push(exp)
                                else: # Should not be reached if PER is True by default
                                    agent.replay_buffer.append(exp)
                        except Exception as e_push:
                            logging.error(f"将 experience 推送到缓冲区时出错: {e_push}");
                            traceback.print_exc()

                        if len(n_step_buffer) >= config.n_step:
                            n_step_buffer.popleft()
                else: # 1-step
                    exp = Experience(state_norm, action, norm_reward, next_state_norm, done, 1, next_lane_idx)
                    try: # Robust push
                        if exp.state is not None and exp.next_state is not None and \
                           not (np.any(np.isnan(exp.state)) or np.any(np.isinf(exp.state))) and \
                           not (np.any(np.isnan(exp.next_state)) or np.any(np.isinf(exp.next_state))) and \
                           np.isfinite(exp.n_step_reward):
                            if config.use_per:
                                agent.replay_buffer.push(exp)
                            else:
                                agent.replay_buffer.append(exp)
                    except Exception as e_push:
                        logging.error(f"将 experience 推送到缓冲区时出错: {e_push}");
                        traceback.print_exc()

                state_norm = next_state_norm

                episode_reward_norm_sum += norm_reward
                step_count += 1
                global_step_count += 1

                if global_step_count >= config.learning_starts and len(agent.replay_buffer) >= config.batch_size:
                    loss = agent.update(global_step_count)
                    if loss is not None:
                        if not np.isfinite(loss):
                            logging.warning(
                                f"回合 {episode} 更新步骤 {agent.train_step_count} 损失无效 ({loss})。跳过记录。")
                        else:
                            episode_loss_sum += loss
                            episode_loss_count += 1
                    # V2: Store current learning rate after update and scheduler step
                    if agent.scheduler:
                        learning_rates_history.append(agent.optimizer.param_groups[0]['lr'])
                    else:
                        learning_rates_history.append(config.initial_learning_rate)


                if env.collision_occurred: # Ensure done is true if env flags collision
                    done = True

                if done and config.use_n_step and len(n_step_buffer) > 0: # Process remaining n-step buffer
                    while len(n_step_buffer) > 0:
                        n_step_actual = len(n_step_buffer)
                        n_step_return_discounted = 0.0
                        for i in range(n_step_actual):
                            n_step_return_discounted += (config.gamma ** i) * n_step_buffer[i].reward
                        s_t = n_step_buffer[0].state
                        a_t = n_step_buffer[0].action
                        s_t_plus_n = n_step_buffer[-1].next_state
                        done_n_step = True # Final transitions are 'done' for this context
                        next_lane_idx_n_step = n_step_buffer[-1].next_lane_index
                        exp = Experience(s_t, a_t, n_step_return_discounted, s_t_plus_n, done_n_step,
                                           n_step_actual, next_lane_idx_n_step)
                        try: # Robust push
                            if exp.state is not None and exp.next_state is not None and \
                               not (np.any(np.isnan(exp.state)) or np.any(np.isinf(exp.state))) and \
                               not (np.any(np.isnan(exp.next_state)) or np.any(np.isinf(exp.next_state))) and \
                               np.isfinite(exp.n_step_reward):
                                if config.use_per:
                                    agent.replay_buffer.push(exp)
                                else:
                                    agent.replay_buffer.append(exp)
                        except Exception as e_push:
                            logging.error(f"将最终 experience 推送到缓冲区时出错: {e_push}");
                            traceback.print_exc()
                        n_step_buffer.popleft()

            all_rewards_sum_norm.append(episode_reward_norm_sum)
            lane_change_counts.append(env.change_lane_count)
            collision_counts.append(1 if env.collision_occurred else 0)
            total_steps_per_episode.append(step_count)
            avg_loss_ep = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0
            avg_losses_per_episode.append(avg_loss_ep)

            current_beta = linear_decay(config.per_beta_start, config.per_beta_end,
                                        config.per_beta_annealing_steps, global_step_count)
            beta_values.append(current_beta)

            avg_window = 20
            if episode >= avg_window:
                rewards_slice = all_rewards_sum_norm[-avg_window:]
                current_avg_reward = np.mean(rewards_slice) if rewards_slice else -float('inf')
                if current_avg_reward > best_avg_reward and global_step_count >= config.learning_starts:
                    best_avg_reward = current_avg_reward
                    best_model_path = os.path.join(models_dir, "best_model.pth")
                    torch.save(agent.policy_net.state_dict(), best_model_path)
                    logging.info(f"\n🎉 新的最佳平均奖励 ({avg_window}回合, 归一化总和): {best_avg_reward:.2f}! 模型已保存。")

            if episode % config.log_interval == 0:
                log_slice_start = max(0, episode - config.log_interval)
                avg_reward_log = np.mean(all_rewards_sum_norm[-config.log_interval:]) if all_rewards_sum_norm else 0
                avg_steps_log = np.mean(total_steps_per_episode[-config.log_interval:]) if total_steps_per_episode else 0
                collision_rate_log = np.mean(collision_counts[-config.log_interval:]) * 100 if collision_counts else 0
                # Avg loss only from episodes where learning actually happened
                valid_losses_log = [l for i, l in enumerate(avg_losses_per_episode) if
                                     i >= log_slice_start and total_steps_per_episode[
                                         i] > 0 and ( # Ensure steps were taken
                                                                                             global_step_count - sum( # Approx check if learning happened
                                                                                         total_steps_per_episode[i + 1:]) >= config.learning_starts)]
                avg_loss_log = np.mean(
                    [l for l in avg_losses_per_episode[-config.log_interval:] if l > 0]) if any(
                    l > 0 for l in avg_losses_per_episode[-config.log_interval:]) else 0.0

                avg_lc_log = np.mean(lane_change_counts[-config.log_interval:]) if lane_change_counts else 0
                buffer_fill = f"{len(agent.replay_buffer)}/{config.replay_buffer_size}" if agent.replay_buffer else "N/A"
                current_lr = agent.optimizer.param_groups[0]['lr'] if agent.optimizer else config.initial_learning_rate


                logging.info(
                    f"\n回合: {episode}/{config.dqn_episodes} | Avg Rew (Norm, {config.log_interval}ep): {avg_reward_log:.2f} "
                    f"| Best Avg ({avg_window}ep): {best_avg_reward:.2f} "
                    f"| Avg Steps: {avg_steps_log:.1f} | Avg LC: {avg_lc_log:.1f} "
                    f"| Coll Rate: {collision_rate_log:.1f}% "
                    f"| Avg Loss: {avg_loss_log:.4f} | Beta: {current_beta:.3f} | LR: {current_lr:.2e} "
                    f"| Buffer: {buffer_fill} | Global Steps: {global_step_count}")

            if episode % config.save_interval == 0:
                periodic_model_path = os.path.join(models_dir, f"model_ep{episode}.pth")
                torch.save(agent.policy_net.state_dict(), periodic_model_path)
                logging.info(f"模型已保存: {periodic_model_path}")

    except KeyboardInterrupt:
        logging.info("\n用户中断训练。")
    except Exception as e:
        logging.error(f"\n训练期间发生致命错误: {e}")
        traceback.print_exc()
    finally:
        logging.info("正在关闭最终环境...")
        env._close()

        if 'agent' in locals() and agent is not None and agent.policy_net is not None:
            last_model_path = os.path.join(models_dir, "last_model.pth")
            torch.save(agent.policy_net.state_dict(), last_model_path)
            logging.info(f"最终模型已保存至: {last_model_path}")

            # --- 绘图 (扩展) ---
            logging.info("--- 生成训练图表 ---")
            plot_window = 20

            try:
                logging.info("正在生成奖励和步数图...")
                fig1, axs1 = plt.subplots(2, 1, figsize=(12, 10), # Increased height for LR plot
                                         sharex=True)
                fig1.suptitle(f"DQN 训练 ({exp_name}): 奖励, 步数", fontsize=16)

                ax_rew = axs1[0]
                ax_rew.set_ylabel(f"{plot_window}回合滚动平均奖励 (归一化)")
                ax_rew.grid(True, linestyle='--')
                if len(all_rewards_sum_norm) >= plot_window:
                    rolling_avg_reward = calculate_rolling_average(all_rewards_sum_norm, plot_window)
                    episode_axis_rolled = np.arange(plot_window - 1, len(all_rewards_sum_norm)) + 1
                    ax_rew.plot(episode_axis_rolled, rolling_avg_reward,
                            label=f'{plot_window}回合滚动平均奖励', color='red', linewidth=2)
                    episode_axis_raw = np.arange(1, len(all_rewards_sum_norm) + 1)
                    ax_rew.plot(episode_axis_raw, all_rewards_sum_norm, label='每回合奖励 (归一化)', color='lightcoral',
                            alpha=0.3, marker='.', linestyle='None')
                    ax_rew.legend(loc='best')
                else:
                    ax_rew.text(0.5, 0.5, '数据不足', horizontalalignment='center',
                            verticalalignment='center', transform=ax_rew.transAxes)

                ax_steps = axs1[1]
                ax_steps.set_xlabel("回合")
                ax_steps.set_ylabel(f"{plot_window}回合滚动平均步数")
                ax_steps.grid(True, linestyle='--')
                if len(total_steps_per_episode) >= plot_window:
                    rolling_avg_steps = calculate_rolling_average(total_steps_per_episode, plot_window)
                    episode_axis_rolled = np.arange(plot_window - 1, len(total_steps_per_episode)) + 1
                    ax_steps.plot(episode_axis_rolled, rolling_avg_steps,
                            label=f'{plot_window}回合滚动平均步数', color='blue', linewidth=2)
                    episode_axis_raw = np.arange(1, len(total_steps_per_episode) + 1)
                    ax_steps.plot(episode_axis_raw, total_steps_per_episode, label='每回合步数', color='lightblue',
                            alpha=0.3, marker='.', linestyle='None')
                    ax_steps.legend(loc='best')
                else:
                    ax_steps.text(0.5, 0.5, '数据不足', horizontalalignment='center',
                            verticalalignment='center', transform=ax_steps.transAxes)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path1 = os.path.join(results_dir, "dqn_training_rewards_steps.png")
                plt.savefig(plot_path1)
                plt.close(fig1)
                logging.info(f"奖励/步数图已保存至: {plot_path1}")
            except Exception as e_plot1:
                logging.error(f"绘制奖励/步数图时出错: {e_plot1}");
                traceback.print_exc()

            try:
                logging.info("正在生成损失和学习率图...")
                fig2, ax2_loss = plt.subplots(1, 1, figsize=(12, 6)) # Separate figure for loss & LR
                fig2.suptitle(f"DQN 训练 ({exp_name}): 损失和学习率", fontsize=16)

                # Loss plot (left y-axis)
                ax2_loss.set_xlabel("更新步骤")
                ax2_loss.set_ylabel("损失 (交叉熵)", color='purple')
                ax2_loss.grid(True, linestyle='--')
                loss_per_step = agent.loss_history
                if loss_per_step:
                    update_axis = np.arange(len(loss_per_step))
                    ax2_loss.plot(update_axis, loss_per_step, label='损失/更新步骤', color='mediumpurple', alpha=0.3,
                            linewidth=0.5)
                    loss_plot_window = max(50, min(1000, len(loss_per_step) // 10))
                    if len(loss_per_step) >= loss_plot_window and loss_plot_window > 0:
                        rolling_loss = calculate_rolling_average(loss_per_step, loss_plot_window)
                        update_axis_rolled = np.arange(loss_plot_window - 1, len(loss_per_step))
                        ax2_loss.plot(update_axis_rolled, rolling_loss,
                                label=f'{loss_plot_window}步滚动平均损失', color='purple', linewidth=1.5)
                    ax2_loss.tick_params(axis='y', labelcolor='purple')
                    ax2_loss.legend(loc='upper left')
                else:
                    ax2_loss.text(0.5, 0.5, '无损失数据', horizontalalignment='center',
                             verticalalignment='center', transform=ax2_loss.transAxes)

                # Learning Rate plot (right y-axis)
                if learning_rates_history: # Only plot if LR history exists
                    ax2_lr = ax2_loss.twinx() # Share the same x-axis
                    ax2_lr.set_ylabel("学习率", color='darkgreen')
                    lr_update_axis = np.arange(len(learning_rates_history)) # Should align with loss update steps
                    ax2_lr.plot(lr_update_axis, learning_rates_history, label='学习率/更新步骤', color='darkgreen', linestyle=':', linewidth=1.5)
                    ax2_lr.tick_params(axis='y', labelcolor='darkgreen')
                    ax2_lr.legend(loc='upper right')
                    # ax2_lr.set_yscale('log') # Optionally use log scale for LR if it changes drastically

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path2 = os.path.join(results_dir, "dqn_training_loss_lr.png")
                plt.savefig(plot_path2)
                plt.close(fig2)
                logging.info(f"损失/学习率图已保存至: {plot_path2}")
            except Exception as e_plot2:
                logging.error(f"绘制损失/学习率图时出错: {e_plot2}");
                traceback.print_exc()


            if config.use_per:
                try:
                    logging.info("正在生成 PER Beta 图...")
                    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
                    fig3.suptitle(f"DQN 训练 ({exp_name}): PER Beta 退火", fontsize=16)
                    ax3.set_xlabel("回合")
                    ax3.set_ylabel("Beta 值")
                    ax3.grid(True, linestyle='--')
                    if beta_values:
                        episode_axis_beta = np.arange(1, len(beta_values) + 1)
                        ax3.plot(episode_axis_beta, beta_values, label='PER Beta', color='green')
                        ax3.legend()
                    else:
                        ax3.text(0.5, 0.5, '无 Beta 数据', horizontalalignment='center',
                                verticalalignment='center', transform=ax3.transAxes)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plot_path3 = os.path.join(results_dir, "dqn_training_beta.png")
                    plt.savefig(plot_path3)
                    plt.close(fig3)
                    logging.info(f"Beta 图已保存至: {plot_path3}")
                except Exception as e_plot3:
                    logging.error(f"绘制 Beta 图时出错: {e_plot3}");
                    traceback.print_exc()

            try:
                logging.info("正在生成碰撞和换道图...")
                fig4, axs4 = plt.subplots(2, 1, figsize=(12, 8),
                                         sharex=True)
                fig4.suptitle(f"DQN 训练 ({exp_name}): 碰撞率和换道次数", fontsize=16)

                ax_coll = axs4[0]
                ax_coll.set_ylabel(f"{plot_window}回合滚动平均碰撞率 (%)")
                ax_coll.grid(True, linestyle='--')
                ax_coll.set_ylim(-5, 105)
                if len(collision_counts) >= plot_window:
                    collision_rate = np.array(collision_counts) * 100
                    rolling_avg_coll = calculate_rolling_average(collision_rate, plot_window)
                    episode_axis_rolled_coll = np.arange(plot_window - 1, len(collision_counts)) + 1
                    ax_coll.plot(episode_axis_rolled_coll, rolling_avg_coll,
                            label=f'{plot_window}回合滚动平均碰撞率', color='black', linewidth=2)
                    episode_axis_raw_coll = np.arange(1, len(collision_counts) + 1)
                    ax_coll.plot(episode_axis_raw_coll, collision_rate, label='每回合碰撞 (0/100)', color='darkgray',
                            alpha=0.3, marker='.', linestyle='None')
                    ax_coll.legend(loc='best')
                else:
                    ax_coll.text(0.5, 0.5, '数据不足', horizontalalignment='center',
                            verticalalignment='center', transform=ax_coll.transAxes)

                ax_lc = axs4[1]
                ax_lc.set_xlabel("回合")
                ax_lc.set_ylabel(f"{plot_window}回合滚动平均换道次数")
                ax_lc.grid(True, linestyle='--')
                if len(lane_change_counts) >= plot_window:
                    rolling_avg_lc = calculate_rolling_average(lane_change_counts, plot_window)
                    episode_axis_rolled_lc = np.arange(plot_window - 1, len(lane_change_counts)) + 1
                    ax_lc.plot(episode_axis_rolled_lc, rolling_avg_lc,
                            label=f'{plot_window}回合滚动平均换道次数', color='orange', linewidth=2)
                    episode_axis_raw_lc = np.arange(1, len(lane_change_counts) + 1)
                    ax_lc.plot(episode_axis_raw_lc, lane_change_counts, label='每回合换道次数', color='moccasin',
                            alpha=0.3, marker='.', linestyle='None')
                    ax_lc.legend(loc='best')
                else:
                    ax_lc.text(0.5, 0.5, '数据不足', horizontalalignment='center',
                            verticalalignment='center', transform=ax_lc.transAxes)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path4 = os.path.join(results_dir, "dqn_training_collisions_lc.png")
                plt.savefig(plot_path4)
                plt.close(fig4)
                logging.info(f"碰撞/换道图已保存至: {plot_path4}")
            except Exception as e_plot4:
                logging.error(f"绘制碰撞/换道图时出错: {e_plot4}");
                traceback.print_exc()


            logging.info("正在保存训练数据...")
            training_data = {
                "episode_rewards_sum_norm": all_rewards_sum_norm,
                "lane_changes": lane_change_counts,
                "collisions": collision_counts,
                "steps_per_episode": total_steps_per_episode,
                "avg_losses_per_episode": avg_losses_per_episode,
                "detailed_loss_history_per_step": agent.loss_history,
                "beta_values_per_episode": beta_values,
                "learning_rates_per_step": learning_rates_history # V2: Save LR history
            }
            data_path = os.path.join(results_dir, f"training_data_{exp_name}.json")
            try:
                def default_serializer(obj):
                    if isinstance(obj, (np.integer, np.int_)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float_)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, (datetime.datetime, datetime.date)):
                        return obj.isoformat()
                    elif isinstance(obj, (int, float, bool, str, list, dict, type(None))):
                        return obj
                    return str(obj) # Fallback

                cleaned_training_data = {}
                for key, value_list in training_data.items():
                    if isinstance(value_list, list):
                        cleaned_list = []
                        for item in value_list:
                            if isinstance(item, (float, np.floating)) and not np.isfinite(item):
                                cleaned_list.append(0.0) # Replace non-finite with 0.0
                            else:
                                cleaned_list.append(default_serializer(item))
                        cleaned_training_data[key] = cleaned_list
                    else: # Should ideally not happen for current data structure
                        cleaned_training_data[key] = default_serializer(value_list)

                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_training_data, f, indent=4, ensure_ascii=False)
                logging.info(f"训练数据已保存至: {data_path}")
            except TypeError as e_type:
                logging.error(f"保存训练数据时发生类型错误: {e_type}")
                import pprint
                logging.error("Problematic data structure snippet (first 10 items per list):")
                # Log instead of print for consistency
                problem_snippet = {k: (v[:10] if isinstance(v, list) else v) for k,v in cleaned_training_data.items()}
                logging.error(pprint.pformat(problem_snippet))
            except Exception as e:
                logging.error(f"保存训练数据时发生未知错误: {e}");
                traceback.print_exc()
        else:
            logging.info("智能体未初始化或无策略网络，无法保存最终模型/数据。")
        logging.info(f"\n DQN 训练 ({exp_name}) 完成。结果已保存在: {results_dir}")


if __name__ == "__main__":
    main()