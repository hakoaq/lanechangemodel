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

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

# --- 导入必要的组件 ---
try:
    from dqn import Config as DQN_Config
    from dqn import QNetwork, NoisyLinear, RunningMeanStd as DQN_RunningMeanStd, SumoEnv as DQN_SumoEnv
    from ppo import Config as PPO_Config
    from ppo import PPO, BehaviorCloningNet
    from ppo import RunningMeanStd as PPO_RunningMeanStd, SumoEnv as PPO_SumoEnv
except ImportError as e:
    print(f"Error importing from dqn.py or ppo.py: {e}")
    print("Please ensure dqn.py and ppo.py are in the same directory or accessible in the Python path.")
    sys.exit(1)

#####################################
# 评估配置                         #
#####################################
class EvalConfig:
    # --- 模型路径 ---
    # 重要提示：请将这些路径更新为您保存的模型文件
    DQN_MODEL_PATH = "dqn.pth" # CHANGE ME
    PPO_MODEL_PATH = "ppo.pth"              # CHANGE ME

    # --- SUMO 配置 ---
    EVAL_SUMO_BINARY = "sumo"  # 评估时使用 GUI 进行可视化: "sumo-gui"
    EVAL_SUMO_CONFIG = "new.sumocfg" # 使用新的配置文件
    EVAL_STEP_LENGTH = 0.2         # 应与训练步长匹配
    EVAL_PORT_RANGE = (8910, 8920) # 使用与训练不同的端口范围

    # --- 评估参数 ---
    EVAL_EPISODES = 100             # 每个模型运行的回合数
    EVAL_MAX_STEPS = 1500          # 每次评估回合的最大步数 (例如，300 秒)
    EVAL_SEED = 42                 # 如果需要，用于可重复性的种子 (应用于 SumoEnv 启动)
    NUM_LANES = 4                  # new.net.xml 中的车道数 (0, 1, 2, 3)
    EGO_INSERTION_DELAY_SECONDS = 300.0 # <<< NEW: 在插入 Ego 之前的延迟（秒）>>>

    # --- 强制换道尝试逻辑 ---
    FORCE_CHANGE_INTERVAL_STEPS = 75 # 每 X 步尝试一次强制换道
    FORCE_CHANGE_MONITOR_STEPS = 15  # 等待/监控换道完成的步数
    FORCE_CHANGE_SUCCESS_DIST = 5.0  # 成功检查的最小横向移动距离

    # --- 归一化 ---
    # 从原始配置加载归一化设置，但在此处管理状态
    # 假设如果各自的配置说使用了归一化，则两个模型都使用了归一化。

#####################################
# 辅助函数 (复制/改编)              #
#####################################
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
    except Exception as e: print(f"警告：终止 SUMO 进程时出错：{e}")
    time.sleep(0.1)

# 使用 DQN 脚本中的 RunningMeanStd (与 PPO 版本相同)
RunningMeanStd = DQN_RunningMeanStd

#####################################
# 评估环境封装                       #
#####################################
# 使用略微修改的 SumoEnv，为保持一致性，继承了 DQN 版本的大部分内容
# 移除了奖励计算复杂性，专注于状态和执行
class EvaluationEnv:
    def __init__(self, eval_config: EvalConfig, sumo_seed: int):
        self.config = eval_config # 使用评估配置
        self.sumo_binary = self.config.EVAL_SUMO_BINARY
        self.config_path = self.config.EVAL_SUMO_CONFIG
        self.step_length = self.config.EVAL_STEP_LENGTH
        # 假设 ego 车辆 ID 与训练文件中的相同
        self.ego_vehicle_id = DQN_Config.ego_vehicle_id
        self.ego_type_id = DQN_Config.ego_type_id
        self.port_range = self.config.EVAL_PORT_RANGE
        self.num_lanes = self.config.NUM_LANES
        self.sumo_seed = sumo_seed
        self.ego_insertion_delay_steps = int(self.config.EGO_INSERTION_DELAY_SECONDS / self.step_length) # <<< NEW: 计算延迟步数 >>>

        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_raw_state = np.zeros(DQN_Config.state_dim) # 假设状态维度相同
        self.current_step = 0
        self.collision_occurred = False # 标志只在检测到真实碰撞时设置
        self.ego_start_pos = None
        self.ego_route_id = "route_E0" # 假设来自 new.rou.xml 的路径 ID

    def _start_sumo(self):
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.port_range[0], self.port_range[1])
        except IOError as e:
             print(f"错误：无法找到可用端口：{e}")
             sys.exit(1)

        sumo_cmd = [
            self.sumo_binary, "-c", self.config_path,
            "--remote-port", str(self.traci_port),
            "--step-length", str(self.step_length),
            "--collision.check-junctions", "true",
            "--collision.action", "warn", # 重要：使用 warn 获取碰撞信息，但让 RL 处理
            "--time-to-teleport", "-1", # 禁用传送
            "--no-warnings", "true",
            "--seed", str(self.sumo_seed) # 使用提供的种子
        ]
        if self.sumo_binary == "sumo-gui":
            sumo_cmd.extend(["--quit-on-end", "false"]) # 仿真结束后保持 GUI 打开

        try:
             stdout_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
             print(f"在端口 {self.traci_port} 上启动 SUMO，种子为 {self.sumo_seed}...")
        except FileNotFoundError:
             print(f"错误：未找到 SUMO 可执行文件 '{self.sumo_binary}'。")
             sys.exit(1)
        except Exception as e:
             print(f"错误：无法启动 SUMO 进程：{e}")
             sys.exit(1)

        connection_attempts = 5
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.0 + attempt * 0.5)
                traci.init(self.traci_port)
                print(f"SUMO TraCI 已连接 (端口: {self.traci_port})。")
                return
            except traci.exceptions.TraCIException:
                if attempt == connection_attempts - 1:
                    print("达到最大 TraCI 连接尝试次数。")
                    self._close()
                    raise ConnectionError(f"无法连接到 SUMO (端口: {self.traci_port})。")
            except Exception as e:
                print(f"连接 TraCI 时发生意外错误：{e}")
                self._close()
                raise ConnectionError(f"连接到 SUMO 时发生未知错误 (端口: {self.traci_port})。")

    def _add_ego_vehicle(self):
        """使用训练配置中的设置将 ego 车辆添加到仿真中"""
        # 检查路径 (假设来自 new.rou.xml 的 'route_E0')
        if self.ego_route_id not in traci.route.getIDList():
            edge_list = list(traci.edge.getIDList())
            first_edge = edge_list[0] if edge_list else None
            if first_edge:
                print(f"警告：未找到路径 '{self.ego_route_id}'。正在从第一条边 '{first_edge}' 创建路径。")
                try:
                    traci.route.add(self.ego_route_id, [first_edge])
                except traci.exceptions.TraCIException as e:
                    raise RuntimeError(f"使用边 '{first_edge}' 添加路径 '{self.ego_route_id}' 失败：{e}")
            else:
                raise RuntimeError(f"未找到路径 '{self.ego_route_id}' 且没有可用边来创建它。")


        # 检查类型
        if self.ego_type_id not in traci.vehicletype.getIDList():
            try:
                traci.vehicletype.copy("car", self.ego_type_id) # 从 new.rou.xml 中的默认 'car' 类型复制
                traci.vehicletype.setParameter(self.ego_type_id, "color", "1,0,0") # 红色
                # 如果需要，应用训练配置中的关键参数 (评估时可选)
                # traci.vehicletype.setParameter(self.ego_type_id, "lcStrategic", "1.0")
            except traci.exceptions.TraCIException as e:
                print(f"警告：为 Ego 类型 '{self.ego_type_id}' 设置参数失败：{e}")

        # 移除残留的 ego
        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"警告：移除残留 Ego 失败：{e}")

        # 添加 ego
        try:
            start_lane = random.choice(range(self.num_lanes)) # 随机起始车道
            traci.vehicle.add(vehID=self.ego_vehicle_id, routeID=self.ego_route_id,
                              typeID=self.ego_type_id, depart="now",
                              departLane=start_lane, departSpeed="max")

            # 等待 ego 出现
            wait_steps = int(2.0 / self.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.ego_vehicle_id in traci.vehicle.getIDList():
                    ego_appeared = True
                    self.ego_start_pos = traci.vehicle.getPosition(self.ego_vehicle_id)
                    print(f"✅ Ego 车辆 '{self.ego_vehicle_id}' 已添加到仿真中。") # 确认信息
                    break
            if not ego_appeared:
                raise RuntimeError(f"Ego 车辆 '{self.ego_vehicle_id}' 在 {wait_steps} 步内未出现。")

        except traci.exceptions.TraCIException as e:
            print(f"错误：添加 Ego 车辆 '{self.ego_vehicle_id}' 失败：{e}")
            raise RuntimeError("添加 Ego 车辆失败。")

    def reset(self) -> np.ndarray:
        """为新的评估回合重置环境"""
        self._close()
        self._start_sumo()

        # <<< CHANGE START: Delayed ego insertion >>>
        print(f"运行仿真 {self.config.EGO_INSERTION_DELAY_SECONDS:.1f} 秒 ({self.ego_insertion_delay_steps} 步) 的延迟...")
        for _ in range(self.ego_insertion_delay_steps):
            try:
                traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                 print(f"延迟期间发生 TraCI 错误：{e}")
                 # 根据错误严重程度决定是继续还是中止
                 if "connection closed" in str(e).lower():
                     raise ConnectionError("SUMO 连接在延迟期间关闭。")
                 # 其他错误可以尝试继续
        print("延迟完成。正在添加 Ego 车辆...")
        # <<< CHANGE END: Delayed ego insertion >>>

        self._add_ego_vehicle()
        self.current_step = 0
        self.collision_occurred = False # 每次重置时确保重置碰撞标志
        self.last_raw_state = np.zeros(DQN_Config.state_dim)

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                raw_state = self._get_raw_state()
                self.last_raw_state = raw_state.copy()
            except traci.exceptions.TraCIException:
                 print("警告：在 reset 中的初始状态获取期间发生 TraCI 异常。")
        else:
             print("警告：在 reset 中的 add/wait 后未立即找到 Ego 车辆。")

        return self.last_raw_state

    # 重用 dqn.py 中的 _get_surrounding_vehicle_info 逻辑 (与 ppo.py 相同)
    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """获取最近车辆的距离和相对速度 (从 dqn.py 复制)"""
        max_dist = DQN_Config.max_distance # 使用配置中的距离
        infos = {
            'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0),
            'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0),
            'right_back': (max_dist, 0.0)
        }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        try:
            ego_road = traci.vehicle.getRoadID(ego_id)
            if not ego_road: return infos # 检查是否在道路上
            num_lanes_on_edge = self.num_lanes # 使用已知车道数
        except traci.exceptions.TraCIException:
            # 如果在获取 ego 道路时出错，返回默认值
            return infos

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                # 首先检查车辆是否在同一条边上
                veh_road = traci.vehicle.getRoadID(veh_id)
                if veh_road != ego_road: continue

                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)

                # 检查车道索引是否有效
                if not (0 <= veh_lane < num_lanes_on_edge): continue

                # 状态的简单纵向距离假设
                dx = veh_pos[0] - ego_pos[0]
                # 横向距离检查也可能有用：dy = veh_pos[1] - ego_pos[1]
                distance = abs(dx) # 使用纵向距离以保持状态表示的一致性

                if distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed # 如果 ego 更快则为正

                # 根据车道和 dx 确定相对位置
                if veh_lane == ego_lane: # 同一车道
                    if dx > 0 and distance < infos['front'][0]: infos['front'] = (distance, rel_speed)
                    # 对于此状态表示，忽略同一车道正后方的车辆
                elif veh_lane == ego_lane - 1: # 左车道
                    if dx > -5 and distance < infos['left_front'][0]: # 检查略微靠后到前方的车辆
                        infos['left_front'] = (distance, rel_speed)
                    elif dx <= -5 and distance < infos['left_back'][0]: # 检查更靠后的车辆
                        infos['left_back'] = (distance, rel_speed)
                elif veh_lane == ego_lane + 1: # 右车道
                     if dx > -5 and distance < infos['right_front'][0]:
                        infos['right_front'] = (distance, rel_speed)
                     elif dx <= -5 and distance < infos['right_back'][0]:
                        infos['right_back'] = (distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue # 如果发生 TraCI 错误，则跳过该车辆
        return infos

    # 重用 dqn.py 中的 _get_raw_state 逻辑
    def _get_raw_state(self) -> np.ndarray:
        """获取当前环境状态 (归一化前的原始值，从 dqn.py 复制)"""
        state = np.zeros(DQN_Config.state_dim, dtype=np.float32)
        ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList():
            return self.last_raw_state # 如果 ego 消失，则返回最后已知的原始状态

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road:
                return self.last_raw_state # 不在道路上

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            num_lanes = self.num_lanes # 使用已知车道数

            # 确保 ego_lane 有效
            if not (0 <= ego_lane < num_lanes):
                 print(f"警告：检测到无效的 ego 车道 {ego_lane}。进行裁剪。")
                 ego_lane = np.clip(ego_lane, 0, num_lanes - 1)

            # 检查换道可能性 (TraCI 在内部处理边界检查)
            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            # --- 原始状态特征 (匹配训练状态) ---
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
                print(f"警告：在原始状态计算中检测到 NaN 或 Inf。使用最后有效的原始状态。")
                return self.last_raw_state

            self.last_raw_state = state.copy() # 存储最新的有效原始状态

        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" in str(e): pass # 接近结束时预期
            else: print(f"警告：获取 {ego_id} 的原始状态时发生 TraCI 错误：{e}。返回最后已知的原始状态。")
            return self.last_raw_state
        except Exception as e:
            print(f"警告：获取 {ego_id} 的原始状态时发生未知错误：{e}。返回最后已知的原始状态。")
            traceback.print_exc()
            return self.last_raw_state

        return state

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """执行动作，返回 (next_raw_state, done)"""
        done = False
        ego_id = self.ego_vehicle_id

        # 在步骤开始时检查 ego 是否仍然存在
        if ego_id not in traci.vehicle.getIDList():
             # 如果在步骤开始时 ego 就消失了，这很可能是一个问题，但我们不会立即将其标记为碰撞。
             # 更有可能的是，它在上一步中到达了终点。
             # 返回最后状态，并将 done 标记为 True。碰撞状态应由上一步决定。
             print(f"警告：Ego {ego_id} 在步骤 {self.current_step} 开始时不存在。")
             return self.last_raw_state, True

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id)
            num_lanes = self.num_lanes

            # 在执行动作之前确保 current_lane 有效
            if not (0 <= current_lane < num_lanes):
                 current_lane = np.clip(current_lane, 0, num_lanes - 1)

            # 1. 执行动作
            if action == 1 and current_lane > 0: # 尝试左转
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
            elif action == 2 and current_lane < (num_lanes - 1): # 尝试右转
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
            # 动作 0: 保持车道 (无需显式操作)

            # 2. 仿真步骤
            traci.simulationStep()
            self.current_step += 1

            # 3. 步骤之后检查状态

            # <<< CHANGE START: Refined Collision/End Detection >>>
            # 检查 SUMO 报告的显式碰撞
            collisions = traci.simulation.getCollisions()
            ego_collided_explicitly = False
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True # 设置真实碰撞标志
                    ego_collided_explicitly = True
                    done = True
                    print(f"💥 检测到涉及 {ego_id} 的碰撞，由 SUMO 报告，在步骤 {self.current_step}")
                    break # 找到碰撞

            # 检查 ego 在步骤后是否存在
            ego_exists_after_step = ego_id in traci.vehicle.getIDList()

            # 如果 ego 消失且 SUMO 未报告碰撞，则假定是正常结束
            if not ego_exists_after_step and not ego_collided_explicitly:
                print(f"ℹ️ Ego {ego_id} 在步骤 {self.current_step} 后消失 (可能到达路径终点)。不视为碰撞。")
                done = True
                # 不要在此处设置 self.collision_occurred = True

            # <<< CHANGE END: Refined Collision/End Detection >>>

            # 获取下一个状态 (原始) - 仅当 ego 存在时；否则使用最后一个已知状态
            if ego_exists_after_step:
                next_raw_state = self._get_raw_state()
            else:
                next_raw_state = self.last_raw_state

            # 检查其他终止条件
            sim_time = traci.simulation.getTime()
            if sim_time >= 3600: done = True # 仿真时间限制 (例如 1 小时)
            if self.current_step >= self.config.EVAL_MAX_STEPS: done = True # 评估步数限制

        except traci.exceptions.TraCIException as e:
            # 检查错误是否与车辆消失有关
            is_not_known_error = "Vehicle '" + ego_id + "' is not known" in str(e)
            if is_not_known_error:
                # 如果 SUMO 未报告碰撞，则可能已正常结束
                if not self.collision_occurred:
                    print(f"ℹ️ Ego {ego_id} 在 TraCI 异常期间消失 (可能已结束)。不视为碰撞。")
                # 如果已经发生碰撞，则 collision_occurred 已设置
            else:
                print(f"错误：在步骤 {self.current_step} 期间发生 TraCI 异常：{e}")
                # 对于其他 TraCI 错误，可能需要将其视为问题/碰撞
                # self.collision_occurred = True # 取消注释以将 TraCI 错误视为碰撞
            done = True
            next_raw_state = self.last_raw_state
        except Exception as e:
            print(f"错误：在步骤 {self.current_step} 期间发生未知异常：{e}")
            traceback.print_exc()
            done = True
            self.collision_occurred = True # 在未知错误时假设发生碰撞
            next_raw_state = self.last_raw_state

        # 返回原始状态和完成标志
        return next_raw_state, done

    def get_vehicle_info(self):
        """获取当前信息，如速度、车道、位置"""
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
            # 如果 ego 不存在，返回默认值
            # 检查 self.collision_occurred 以确定原因
            # dist_traveled = math.dist(self.last_raw_state[?], self.ego_start_pos) ? # 获取最后位置可能很复杂
            last_dist = 0.0
            if hasattr(self, 'last_ego_pos') and self.last_ego_pos and self.ego_start_pos:
                last_dist = math.dist(self.last_ego_pos, self.ego_start_pos)
            elif self.ego_start_pos: # Fallback if last_ego_pos wasn't stored
                # Try getting from last_raw_state if possible, otherwise 0
                 pass # Difficult to reliably get final position from raw state vector

            return {"speed": 0, "lane": -1, "pos": getattr(self, 'last_ego_pos', (0,0)), "dist": last_dist}


    def _close(self):
        """关闭 SUMO 实例"""
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
                except Exception as e: print(f"警告：SUMO 终止期间出错：{e}")
                self.sumo_process = None
                self.traci_port = None
                time.sleep(0.1)
        else:
            self.traci_port = None

#####################################
# 模型加载和动作选择                   #
#####################################

def load_model(model_path: str, model_type: str, config: Any, device: torch.device) -> nn.Module:
    """加载训练好的模型 (DQN 或 PPO)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    if model_type == 'dqn':
        if not isinstance(config, DQN_Config):
             print(f"调试：预期为 DQN_Config，但得到类型：{type(config)}")
             raise TypeError("为 DQN 模型提供的配置不是 DQN_Config 的实例")
        model = QNetwork(config.state_dim, config.action_dim, config.hidden_size, config).to(device)
    elif model_type == 'ppo':
        if not isinstance(config, PPO_Config):
             print(f"调试：预期为 PPO_Config，但得到类型：{type(config)}")
             raise TypeError("为 PPO 模型提供的配置不是 PPO_Config 的实例")
        model = PPO(config.state_dim, config.action_dim, config.hidden_size).to(device)
    else:
        raise ValueError(f"未知的模型类型：{model_type}")

    try:
        # 显式禁用 strict loading，以防 BC 模型权重与 PPO actor 不完全匹配 (尽管它们应该匹配)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True if model_type=='dqn' else False)
        model.eval() # 设置为评估模式
        print(f"成功从以下位置加载 {model_type.upper()} 模型：{model_path}")
        return model
    except Exception as e:
        print(f"从 {model_path} 加载模型 state_dict 时出错：{e}")
        raise

def normalize_state(state_raw: np.ndarray, normalizer: Optional[RunningMeanStd], clip_val: float) -> np.ndarray:
    """使用提供的归一化器实例归一化状态"""
    if normalizer:
        # 不在评估期间更新归一化器统计信息 - 使用训练结束时加载的统计信息
        # 使用冻结的 normalizer
        mean = normalizer.mean
        std = normalizer.std + 1e-8 # 添加 epsilon 防止除以零

        norm_state = (state_raw - mean) / std
        norm_state = np.clip(norm_state, -clip_val, clip_val)
        return norm_state.astype(np.float32)
    else:
        return state_raw.astype(np.float32) # 如果没有归一化器，则返回原始状态

def get_dqn_action(model: QNetwork, state_norm: np.ndarray, current_lane_idx: int, config: DQN_Config, device: torch.device) -> int:
    """从 DQN 模型 (C51/Noisy 感知) 获取动作"""
    model.eval() # 确保评估模式

    # 如果使用 Noisy Nets，则重置噪声
    if config.use_noisy_nets:
        for module in model.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        action_probs = model(state_tensor) # 输出：[1, action_dim, num_atoms]
        support = torch.linspace(config.v_min, config.v_max, config.num_atoms).to(device)
        expected_q_values = (action_probs * support).sum(dim=2) # [1, action_dim]

        # 动作屏蔽
        q_values_masked = expected_q_values.clone()
        if current_lane_idx == 0:
            q_values_masked[0, 1] = -float('inf') # 不能从车道 0 左转
        if current_lane_idx >= EvalConfig.NUM_LANES - 1:
            q_values_masked[0, 2] = -float('inf') # 不能从最后一条车道右转

        action = q_values_masked.argmax().item()
    return action

def get_ppo_action(model: PPO, state_norm: np.ndarray, current_lane_idx: int, device: torch.device) -> int:
    """从 PPO actor 获取确定性动作"""
    model.eval() # 确保评估模式
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        # 从 actor 部分获取动作概率
        action_probs = model.get_action_probs(state_tensor) # [1, action_dim]

        # 动作屏蔽
        probs_masked = action_probs.clone()
        if current_lane_idx == 0:
            probs_masked[0, 1] = -float('inf') # 通过将概率设置为 ~0 来有效屏蔽
        if current_lane_idx >= EvalConfig.NUM_LANES - 1:
            probs_masked[0, 2] = -float('inf')

        # 屏蔽后选择概率最高的动作
        action = probs_masked.argmax().item()
    return action


#####################################
# 评估回合运行器                    #
#####################################

EpisodeResult = namedtuple('EpisodeResult', [
    'steps', 'collided', 'avg_speed', 'total_dist',
    'forced_attempts', 'forced_agreed', 'forced_executed_safe', 'forced_executed_collision',
    'model_lane_changes'
])

def evaluate_episode(
    model: nn.Module,
    model_type: str, # 'dqn' 或 'ppo'
    env: EvaluationEnv,
    config: Any, # DQN_Config 或 PPO_Config
    obs_normalizer: Optional[RunningMeanStd], # 初始（冻结的）归一化器状态
    device: torch.device,
    eval_config: EvalConfig
) -> Tuple[EpisodeResult, Optional[RunningMeanStd]]: # 返回结果，不再需要更新 normalizer
    """为给定模型运行单个评估回合"""

    state_raw = env.reset() # 返回原始状态
    # normalizer 在评估期间是冻结的，不需要副本
    current_obs_normalizer = obs_normalizer

    done = False
    step_count = 0
    speeds = []
    model_lane_changes = 0 # 计算模型动作 != 0 启动的换道次数

    # 强制换道跟踪
    forced_attempts = 0
    forced_agreed = 0           # 模型选择了期望的动作
    forced_executed_safe = 0    # 模型同意，换道完成，无碰撞
    forced_executed_collision = 0 # 模型同意，尝试换道，发生碰撞
    monitoring_change = False
    monitor_steps_left = 0
    monitor_target_action = -1
    monitor_start_lane = -1
    monitor_start_pos = None

    last_valid_vehicle_info = env.get_vehicle_info() # 存储最后有效信息以防 ego 消失

    while not done and step_count < eval_config.EVAL_MAX_STEPS:
        # 1. 归一化状态 (使用冻结的 normalizer)
        clip_value = config.obs_norm_clip if hasattr(config, 'obs_norm_clip') else 5.0 # 从相应配置获取
        state_norm = normalize_state(state_raw, current_obs_normalizer, clip_value)

        # 2. 获取当前车道并决定强制换道尝试
        # 使用原始状态中的车道索引
        if not np.any(np.isnan(state_raw)) and len(state_raw) > 1:
             current_lane_idx = int(round(state_raw[1])) # 索引 1 是车道索引
             current_lane_idx = np.clip(current_lane_idx, 0, eval_config.NUM_LANES - 1)
        else:
             # 如果状态无效，则回退到上一个已知有效车道
             current_lane_idx = last_valid_vehicle_info['lane']
             if current_lane_idx < 0: current_lane_idx = 0 # 最终回退

        can_go_left = current_lane_idx > 0
        can_go_right = current_lane_idx < (eval_config.NUM_LANES - 1)
        target_action = -1 # -1 表示无强制动作

        # 检查是否正在监控先前的尝试
        if monitoring_change:
            monitor_steps_left -= 1
            current_vehicle_info = env.get_vehicle_info()
            current_pos = current_vehicle_info['pos']
            current_lane_after_step = current_vehicle_info['lane']

            # 检查成功的物理变化 (显著的横向移动或车道索引变化)
            # 仅当 current_pos 有效时才检查
            lateral_dist = 0.0
            if current_pos and monitor_start_pos:
                lateral_dist = abs(current_pos[1] - monitor_start_pos[1])
            lane_changed_physically = (current_lane_after_step >= 0 and current_lane_after_step != monitor_start_lane)

            # 检查完成标准
            if lane_changed_physically or lateral_dist > eval_config.FORCE_CHANGE_SUCCESS_DIST:
                if env.collision_occurred: # 检查监控 *期间* 是否发生碰撞
                    forced_executed_collision += 1
                    print(f"⚠️ 强制换道 ({monitor_target_action}) 已同意，但导致了碰撞。")
                else:
                    forced_executed_safe += 1
                    print(f"✅ 强制换道 ({monitor_target_action}) 成功执行。")
                monitoring_change = False # 停止监控此尝试
            elif monitor_steps_left <= 0:
                print(f"❌ 强制换道 ({monitor_target_action}) 已同意，但超时 (未执行)。")
                monitoring_change = False # 超时
            elif env.collision_occurred: # 在完成之前发生碰撞
                 forced_executed_collision += 1
                 print(f"⚠️ 强制换道 ({monitor_target_action}) 已同意，但在完成前导致了碰撞。")
                 monitoring_change = False

        # 如果当前未监控，则触发新的强制尝试
        if not monitoring_change and step_count > 0 and step_count % eval_config.FORCE_CHANGE_INTERVAL_STEPS == 0:
            if can_go_left and can_go_right:
                target_action = random.choice([1, 2]) # 如果两者都可能，则随机选择
            elif can_go_left:
                target_action = 1 # 目标左转
            elif can_go_right:
                target_action = 2 # 目标右转

            if target_action != -1:
                forced_attempts += 1
                print(f"\n--- 步骤 {step_count}: 触发强制换道尝试 (目标动作: {target_action}) ---")


        # 3. 获取模型动作
        if model_type == 'dqn':
            action = get_dqn_action(model, state_norm, current_lane_idx, config, device)
        elif model_type == 'ppo':
            action = get_ppo_action(model, state_norm, current_lane_idx, device)
        else:
            action = 0 # 回退

        # 4. 处理强制换道逻辑
        if target_action != -1:
            print(f"   - 模型选择动作: {action}")
            if action == target_action:
                forced_agreed += 1
                print(f"   - 模型同意强制动作 {target_action}。执行并监控...")
                monitoring_change = True
                monitor_steps_left = eval_config.FORCE_CHANGE_MONITOR_STEPS
                monitor_target_action = target_action
                monitor_start_lane = current_lane_idx
                # 获取监控开始时的位置
                start_info = env.get_vehicle_info()
                if start_info['lane'] != -1: # 仅当 ego 有效时存储
                    monitor_start_pos = start_info['pos']
                else:
                    monitor_start_pos = None # 如果 ego 消失，则无法获取位置
            else:
                print(f"   - 模型不同意 (选择了 {action})。执行模型的选择，不监控。")
                # 继续执行模型选择的动作，不监控

        # 计算模型启动的换道次数 (仅当未监控成功的强制换道时)
        if action != 0 and not monitoring_change:
             model_lane_changes += 1

        # 5. 步骤环境
        next_state_raw, done = env.step(action)

        # 更新状态和指标
        state_raw = next_state_raw
        step_count += 1
        vehicle_info = env.get_vehicle_info()
        # 仅当 ego 存在时记录速度 (速度 > 0)
        if vehicle_info['lane'] != -1:
             speeds.append(vehicle_info['speed'])
             last_valid_vehicle_info = vehicle_info # 更新最后有效信息

        # 如果发生碰撞，立即停止监控任何强制换道
        if env.collision_occurred:
            done = True # 确保循环终止
            if monitoring_change:
                 # 检查碰撞是否发生在监控期间
                 # 注意：env.collision_occurred 在发生碰撞的回合中变为 True
                 # 因此，这将在发生碰撞的 *下一个* 步骤检查中捕获
                 forced_executed_collision += 1 # 假设碰撞相关
                 print(f"⚠️ 强制换道 ({monitor_target_action}) 监控被碰撞中断。")
                 monitoring_change = False


    # 回合结束
    avg_speed = np.mean(speeds) if speeds else 0.0
    # 使用最后有效信息获取总距离
    total_dist = last_valid_vehicle_info['dist']
    # 使用环境的最终碰撞状态
    collided = env.collision_occurred # 这是由 step 函数中的真实碰撞设置的

    # 处理在回合结束时监控仍在进行的情况 (超时)
    if monitoring_change:
        print(f"❌ 强制换道 ({monitor_target_action}) 监控在回合结束时仍在进行 (超时)。")

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

    # 不再需要返回 normalizer
    return result, None


#####################################
# 主评估脚本                       #
#####################################
def main():
    eval_config = EvalConfig()
    dqn_train_config = DQN_Config()
    ppo_train_config = PPO_Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # <<< CHANGE START: Output directory >>>
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出将保存在: {output_dir}")
    # <<< CHANGE END: Output directory >>>

    # --- 文件检查 ---
    if not os.path.exists(eval_config.DQN_MODEL_PATH):
        print(f"错误：未找到 DQN 模型路径：{eval_config.DQN_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(eval_config.PPO_MODEL_PATH):
        print(f"错误：未找到 PPO 模型路径：{eval_config.PPO_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(eval_config.EVAL_SUMO_CONFIG):
        print(f"错误：未找到评估 SUMO 配置：{eval_config.EVAL_SUMO_CONFIG}")
        try:
            with open(eval_config.EVAL_SUMO_CONFIG, 'r') as f:
                content = f.read()
                if 'new.net.xml' not in content: print("警告：sumocfg 中未提及 'new.net.xml'？")
                if 'new.rou.xml' not in content: print("警告：sumocfg 中未提及 'new.rou.xml'？")
            if not os.path.exists("new.net.xml"): print("警告：未找到 new.net.xml。")
            if not os.path.exists("new.rou.xml"): print("警告：未找到 new.rou.xml。")
        except Exception as e:
            print(f"无法读取 SUMO 配置：{e}")
        sys.exit(1)


    # --- 加载模型 ---
    print("\n--- 加载模型 ---")
    dqn_model = load_model(eval_config.DQN_MODEL_PATH, 'dqn', dqn_train_config, device)
    ppo_model = load_model(eval_config.PPO_MODEL_PATH, 'ppo', ppo_train_config, device)

    # --- 初始化归一化器 ---
    # 从训练检查点加载，此处不创建新的。
    # 假设归一化器状态与模型一起保存，或者通过其他方式加载。
    # 这里我们创建它们，但它们不会在评估期间更新。
    # 注意：实际中，您需要加载与保存的模型相对应的归一化器统计信息！
    # 为了演示，我们基于配置创建它们。
    dqn_obs_normalizer = RunningMeanStd(shape=(dqn_train_config.state_dim,)) if dqn_train_config.normalize_observations else None
    ppo_obs_normalizer = RunningMeanStd(shape=(ppo_train_config.state_dim,)) if ppo_train_config.normalize_observations else None
    print("初始化（冻结的）归一化器以供评估使用。")
    # 在真实场景中：加载保存的 normalizer 状态


    # --- 初始化环境 ---
    base_seed = eval_config.EVAL_SEED
    env = EvaluationEnv(eval_config, base_seed) # 种子将在循环内部更新

    # --- 运行评估 ---
    print(f"\n--- 运行评估 ({eval_config.EVAL_EPISODES} 个回合/模型) ---")
    dqn_results_list: List[EpisodeResult] = []
    ppo_results_list: List[EpisodeResult] = []

    for i in tqdm(range(eval_config.EVAL_EPISODES), desc="评估回合"):
        episode_seed = base_seed + i
        print(f"\n--- 回合 {i+1}/{eval_config.EVAL_EPISODES} (种子: {episode_seed}) ---")

        # --- 评估 DQN ---
        print("评估 DQN...")
        env.sumo_seed = episode_seed # 设置此回合的种子
        try:
            dqn_result, _ = evaluate_episode(
                dqn_model, 'dqn', env, dqn_train_config, dqn_obs_normalizer, device, eval_config
            )
            dqn_results_list.append(dqn_result)
            print(f"DQN 回合 {i+1} 结果: 步数={dqn_result.steps}, 碰撞={dqn_result.collided}, 平均速度={dqn_result.avg_speed:.2f}")
            print(f"  强制换道: 尝试={dqn_result.forced_attempts}, 同意={dqn_result.forced_agreed}, 安全执行={dqn_result.forced_executed_safe}, 碰撞执行={dqn_result.forced_executed_collision}")
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
            print(f"DQN 评估回合 {i+1} 期间出错：{e}")
            if isinstance(e, KeyboardInterrupt): break
            env._close() # 确保清理
            time.sleep(1)
        except Exception as e_other:
            print(f"DQN 评估回合 {i+1} 期间发生意外错误：{e_other}")
            traceback.print_exc()
            env._close(); time.sleep(1)


        # --- 评估 PPO ---
        print("\n评估 PPO...")
        env.sumo_seed = episode_seed # 为 PPO 使用相同的种子以实现公平性
        try:
            ppo_result, _ = evaluate_episode(
                ppo_model, 'ppo', env, ppo_train_config, ppo_obs_normalizer, device, eval_config
            )
            ppo_results_list.append(ppo_result)
            print(f"PPO 回合 {i+1} 结果: 步数={ppo_result.steps}, 碰撞={ppo_result.collided}, 平均速度={ppo_result.avg_speed:.2f}")
            print(f"  强制换道: 尝试={ppo_result.forced_attempts}, 同意={ppo_result.forced_agreed}, 安全执行={ppo_result.forced_executed_safe}, 碰撞执行={ppo_result.forced_executed_collision}")
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
            print(f"PPO 评估回合 {i+1} 期间出错：{e}")
            if isinstance(e, KeyboardInterrupt): break
            env._close() # 确保清理
            time.sleep(1)
        except Exception as e_other:
            print(f"PPO 评估回合 {i+1} 期间发生意外错误：{e_other}")
            traceback.print_exc()
            env._close(); time.sleep(1)


    # 在所有回合结束后关闭环境
    env._close()
    print("\n--- 评估完成 ---")

    # --- 聚合和比较结果 ---
    if not dqn_results_list or not ppo_results_list:
        print("未收集到评估结果。正在退出。")
        return

    results = {'dqn': {}, 'ppo': {}}
    metrics_definitions = [
        # (key in results dict, display name, format string, higher is better?)
        ('avg_steps', '平均步数', '.1f', True),
        ('std_steps', '步数标准差', '.1f', False),
        ('avg_speed', '平均速度 (米/秒)', '.2f', True),
        ('std_speed', '速度标准差 (米/秒)', '.2f', False),
        ('avg_dist', '平均距离 (米)', '.1f', True),
        ('collision_rate', '碰撞率 (%)', '.1f', False),
        ('avg_model_lc', '模型发起的平均换道次数', '.1f', None), # 中性指标
        ('total_forced_attempts', '强制换道尝试总数', 'd', None),
        ('forced_agreement_rate', '强制换道同意率 (%)', '.1f', True),
        ('forced_execution_success_rate', '强制换道执行成功率 (%)', '.1f', True),
        ('forced_execution_collision_rate', '强制换道执行碰撞率 (%)', '.1f', False),
    ]

    for model_key, results_list in [('dqn', dqn_results_list), ('ppo', ppo_results_list)]:
        total_episodes = len(results_list)
        results[model_key]['total_episodes'] = total_episodes

        # 基本指标
        results[model_key]['avg_steps'] = np.mean([r.steps for r in results_list])
        results[model_key]['std_steps'] = np.std([r.steps for r in results_list])
        results[model_key]['collision_rate'] = np.mean([r.collided for r in results_list]) * 100
        results[model_key]['avg_speed'] = np.mean([r.avg_speed for r in results_list])
        results[model_key]['std_speed'] = np.std([r.avg_speed for r in results_list])
        results[model_key]['avg_dist'] = np.mean([r.total_dist for r in results_list])
        results[model_key]['avg_model_lc'] = np.mean([r.model_lane_changes for r in results_list]) # 模型启动的平均换道次数

        # 强制换道指标
        total_forced_attempts = sum(r.forced_attempts for r in results_list)
        total_forced_agreed = sum(r.forced_agreed for r in results_list)
        total_forced_executed_safe = sum(r.forced_executed_safe for r in results_list)
        total_forced_executed_collision = sum(r.forced_executed_collision for r in results_list)

        results[model_key]['total_forced_attempts'] = total_forced_attempts
        # 同意率：模型选择目标动作的强制尝试百分比
        results[model_key]['forced_agreement_rate'] = (total_forced_agreed / total_forced_attempts * 100) if total_forced_attempts > 0 else 0
        # 执行成功率：*同意的* 尝试中安全完成的百分比
        results[model_key]['forced_execution_success_rate'] = (total_forced_executed_safe / total_forced_agreed * 100) if total_forced_agreed > 0 else 0
        # 执行碰撞率：*同意的* 尝试中导致碰撞的百分比 (监控期间)
        results[model_key]['forced_execution_collision_rate'] = (total_forced_executed_collision / total_forced_agreed * 100) if total_forced_agreed > 0 else 0

    # --- 打印和保存文本比较 ---
    # <<< CHANGE START: Text Output >>>
    print("\n--- 结果比较 (文本) ---")
    comparison_lines = []
    header1 = f"{'指标':<35} | {'DQN':<20} | {'PPO':<20}"
    header2 = "-" * (35 + 20 + 20 + 5)
    comparison_lines.append(header1)
    comparison_lines.append(header2)
    print(header1)
    print(header2)

    for key, name, fmt, _ in metrics_definitions:
        dqn_val = results['dqn'].get(key, 'N/A')
        ppo_val = results['ppo'].get(key, 'N/A')
        dqn_str = format(dqn_val, fmt) if isinstance(dqn_val, (int, float)) else str(dqn_val)
        ppo_str = format(ppo_val, fmt) if isinstance(ppo_val, (int, float)) else str(ppo_val)
        line = f"{name:<35} | {dqn_str:<20} | {ppo_str:<20}"
        comparison_lines.append(line)
        print(line)
    comparison_lines.append(header2)
    print(header2)

    # 保存到文件
    text_results_filename = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    try:
        with open(text_results_filename, 'w', encoding='utf-8') as f:
            f.write(f"评估运行时间: {timestamp}\n")
            f.write(f"评估回合数: {eval_config.EVAL_EPISODES}\n")
            f.write(f"最大步数/回合: {eval_config.EVAL_MAX_STEPS}\n")
            f.write(f"DQN 模型: {eval_config.DQN_MODEL_PATH}\n")
            f.write(f"PPO 模型: {eval_config.PPO_MODEL_PATH}\n")
            f.write("\n--- 结果比较 ---\n")
            f.write("\n".join(comparison_lines))
        print(f"文本结果摘要已保存至: {text_results_filename}")
    except Exception as e:
        print(f"保存文本结果时出错：{e}")
    # <<< CHANGE END: Text Output >>>


    # --- 生成单独的图表 ---
    # <<< CHANGE START: Separate Plots >>>
    print("\n--- 生成单独的对比图表 ---")
    models = ['DQN', 'PPO'] # 保持算法名称为英文

    plot_metrics = [
        # (key, title, ylabel, filename_suffix)
        ('forced_agreement_rate', '强制换道: 模型同意率', '比率 (%)', 'forced_agreement_rate'),
        ('forced_execution_success_rate', '强制换道: 执行成功率\n(同意换道的百分比)', '比率 (%)', 'forced_exec_success_rate'),
        ('collision_rate', '总碰撞率', '比率 (%)', 'collision_rate'),
        ('avg_speed', '平均速度', '速度 (米/秒)', 'avg_speed'),
        ('avg_steps', '每回合平均步数', '步数', 'avg_steps'),
        ('avg_model_lc', '模型发起的平均换道次数', '次数', 'avg_model_lc'),
        # 可以根据需要添加 std dev 的图表
        # ('std_speed', '速度标准差', '速度 (米/秒)', 'std_speed'),
        # ('std_steps', '步数标准差', '步数', 'std_steps'),
    ]

    for key, title, ylabel, fname_suffix in plot_metrics:
        plt.figure(figsize=(6, 5)) # 为每个图表创建新图形
        plt.title(f'{title} (DQN vs PPO)', fontsize=14)

        values = [results['dqn'].get(key, 0), results['ppo'].get(key, 0)] # 获取两个模型的值
        colors = ['skyblue', 'lightcoral']

        # 如果有标准差数据，则绘制带误差棒的条形图
        if key.startswith('avg_') and f'std_{key[4:]}' in results['dqn'] and f'std_{key[4:]}' in results['ppo']:
            std_devs = [results['dqn'][f'std_{key[4:]}'], results['ppo'][f'std_{key[4:]}']]
            plt.bar(models, values, yerr=std_devs, capsize=5, color=colors)
            # 在条形图顶部添加值文本 (考虑误差棒)
            for i, v in enumerate(values):
                 plt.text(i, v + std_devs[i] * 0.5 + max(values)*0.02, f"{v:.2f}" if isinstance(v, float) else f"{v}", ha='center')
        else:
            # 绘制简单条形图
            plt.bar(models, values, color=colors)
            # 在条形图顶部添加值文本
            for i, v in enumerate(values):
                 plt.text(i, v + max(values)*0.02, f"{v:.1f}%" if '%' in ylabel else (f"{v:.2f}" if isinstance(v, float) else f"{v}"), ha='center')


        plt.ylabel(ylabel)
        # 调整 y 轴限制以获得更好的可视化效果
        if '%' in ylabel:
             plt.ylim(0, max(values + [1]) * 1.1) # 稍微高于最大百分比
             if max(values) > 90: plt.ylim(0, 105) # 如果接近 100%，则设置为 105
        elif max(values) > 0 :
             plt.ylim(0, max(values) * 1.2) # 增加 20% 的顶部空间
        else:
             plt.ylim(min(values)-1, max(values)+1) # 如果值为负或零

        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        # 保存图表到输出目录
        plot_filename = os.path.join(output_dir, f"plot_{fname_suffix}_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"对比图表已保存至: {plot_filename}")
        plt.close() # 关闭当前图形以释放内存

    # <<< CHANGE END: Separate Plots >>>


    # --- 保存 JSON 数据 ---
    data_filename = os.path.join(output_dir, f"evaluation_data_{timestamp}.json")
    # 添加原始回合数据以供潜在的更深入分析
    results['dqn']['raw_results'] = [r._asdict() for r in dqn_results_list]
    results['ppo']['raw_results'] = [r._asdict() for r in ppo_results_list]
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            # 如果有 numpy 类型混入，则使用自定义编码器 (对于 namedtuple 应该不需要)
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        print(f"比较数据已保存至: {data_filename}")
    except Exception as e:
        print(f"保存比较数据时出错：{e}")

if __name__ == "__main__":
    main()