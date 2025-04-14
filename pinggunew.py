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
    # Import necessary classes from training scripts
    from dqn import Config as DQN_Config_Base
    from dqn import QNetwork, NoisyLinear, RunningMeanStd as DQN_RunningMeanStd
    from ppo import Config as PPO_Config_Base
    from ppo import PPO, BehaviorCloningNet # PPO needs BC net only if loading weights from BC init
    from ppo import RunningMeanStd as PPO_RunningMeanStd

    # NOTE: We import the base classes, then create INSTANCES of them.
    # The deepcopy in the original code was not necessary here.

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
    # 重要提示：请将这些路径更新为您保存的模型文件
    DQN_MODEL_PATH = "dqn.pth" # CHANGE ME - Example path
    PPO_MODEL_PATH = "ppo.pth" # CHANGE ME - Example path

    # --- SUMO 配置 ---
    EVAL_SUMO_BINARY = "sumo"  # 评估时使用 GUI 进行可视化: "sumo-gui"
    EVAL_SUMO_CONFIG = "new.sumocfg" # 使用新的配置文件
    EVAL_STEP_LENGTH = 0.2         # 应与训练步长匹配
    EVAL_PORT_RANGE = (8910, 8920) # 使用与训练不同的端口范围

    # --- 评估参数 ---
    EVAL_EPISODES = 100            # 每个模型运行的回合数 (reduced for quicker testing, maybe increase back to 100)
    EVAL_MAX_STEPS = 1500          # 每次评估回合的最大步数 (e.g., 300 seconds)
    EVAL_SEED = 42                 # 如果需要，用于可重复性的种子 (应用于 SumoEnv 启动)
    NUM_LANES = 4                  # new.net.xml 中的车道数 (0, 1, 2, 3)
    EGO_INSERTION_DELAY_SECONDS = 10.0 # Reduced delay for quicker eval start <<< Adjusted delay >>>

    # --- 强制换道尝试逻辑 ---
    FORCE_CHANGE_INTERVAL_STEPS = 75 # 每 X 步尝试一次强制换道 (15 seconds at 0.2 step)
    FORCE_CHANGE_MONITOR_STEPS = 15  # 等待/监控换道完成的步数 (3 seconds)
    FORCE_CHANGE_SUCCESS_DIST = 5.0  # 成功检查的最小横向移动距离

    # --- 归一化 ---
    # Normalizer state should ideally be loaded alongside the models.
    # For this script, we assume the train configs correctly state if norm was used.
    # We will re-create the normalizer objects but NOT update them during eval.
    # !! In a real scenario, load the saved normalizer stats !!

    # --- Ego Vehicle ID & Type for Evaluation ---
    EVAL_EGO_ID = "eval_ego"
    EVAL_EGO_TYPE = "car_ego_eval" # Use a distinct type ID for eval

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

# Use RunningMeanStd from one of the training scripts (they should be identical)
# Using DQN version here, but PPO's is the same structure
RunningMeanStd = DQN_RunningMeanStd

# --- Helper for Plotting Rolling Average ---
def calculate_rolling_average(data, window):
    if len(data) < window:
        return np.array([]) # Not enough data for a full window
    # Ensure data is numpy array of floats for convolve
    data_np = np.array(data, dtype=float)
    return np.convolve(data_np, np.ones(window) / window, mode='valid')


#####################################
# 评估环境封装                       #
#####################################
# Using slightly modified SumoEnv, inheriting most from DQN version for consistency
# Stripped out reward calculation complexity, focus on state and execution
class EvaluationEnv:
    def __init__(self, eval_config: EvalConfig, dqn_train_config: DQN_Config_Base, sumo_seed: int):
        self.eval_config = eval_config # Use eval config
        self.dqn_train_config = dqn_train_config # Need this for state dim, max distance etc.
        self.sumo_binary = self.eval_config.EVAL_SUMO_BINARY
        self.config_path = self.eval_config.EVAL_SUMO_CONFIG
        self.step_length = self.eval_config.EVAL_STEP_LENGTH
        self.ego_vehicle_id = self.eval_config.EVAL_EGO_ID # Use eval ID
        self.ego_type_id = self.eval_config.EVAL_EGO_TYPE   # Use eval type ID
        self.port_range = self.eval_config.EVAL_PORT_RANGE
        self.num_lanes = self.eval_config.NUM_LANES
        self.sumo_seed = sumo_seed
        self.ego_insertion_delay_steps = int(self.eval_config.EGO_INSERTION_DELAY_SECONDS / self.step_length)

        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_raw_state = np.zeros(self.dqn_train_config.state_dim) # Use state dim from train config
        self.current_step = 0
        self.collision_occurred = False # Flag set only on actual collision detection
        self.ego_start_pos = None
        self.ego_route_id = "route_E0" # Assuming route ID from new.rou.xml
        self.last_ego_pos = None # Store last known position

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
            "--collision.action", "warn", # Important: use warn to get collision info but let RL handle
            "--time-to-teleport", "-1", # Disable teleporting
            "--no-warnings", "true",
            "--seed", str(self.sumo_seed) # Use provided seed
        ]

        try:
             stdout_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             stderr_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
             self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
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
        """Add the ego vehicle to the simulation with evaluation-specific type/color."""
        if self.ego_route_id not in traci.route.getIDList():
            try: traci.route.add(self.ego_route_id, ["E0"])
            except traci.exceptions.TraCIException as e:
                 edge_list = list(traci.edge.getIDList()); first_edge = edge_list[0] if edge_list else None
                 if first_edge:
                     print(f"警告：未找到路径 '{self.ego_route_id}' 或边 'E0'。正在从第一条边 '{first_edge}' 创建路径。")
                     try: traci.route.add(self.ego_route_id, [first_edge])
                     except traci.exceptions.TraCIException as e_add: raise RuntimeError(f"使用边 '{first_edge}' 添加路径 '{self.ego_route_id}' 失败：{e_add}")
                 else: raise RuntimeError(f"未找到路径 '{self.ego_route_id}' 且没有可用边来创建它。")

        if self.ego_type_id not in traci.vehicletype.getIDList():
            base_type = "car" # Copy from the base 'car' type defined in new.rou.xml
            if base_type not in traci.vehicletype.getIDList():
                 print(f"错误：未找到基础车辆类型 '{base_type}'。无法创建评估Ego类型。")
                 if "passenger" in traci.vehicletype.getIDList(): base_type = "passenger"
                 else: raise RuntimeError("Neither 'car' nor 'passenger' vType found.")
            try:
                traci.vehicletype.copy(base_type, self.ego_type_id)
                # *** SET EVALUATION COLOR (e.g., Blue) ***
                traci.vehicletype.setParameter(self.ego_type_id, "color", "0,0,1")
                print(f"为评估创建了 Ego 类型 '{self.ego_type_id}'，颜色设置为蓝色。")
            except traci.exceptions.TraCIException as e: print(f"警告：为 Ego 类型 '{self.ego_type_id}' 设置参数失败：{e}")

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try: traci.vehicle.remove(self.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e: print(f"警告：移除残留 Ego '{self.ego_vehicle_id}' 失败：{e}")

        try:
            start_lane = random.choice(range(self.num_lanes)) # Random start lane
            traci.vehicle.add(vehID=self.ego_vehicle_id, routeID=self.ego_route_id, typeID=self.ego_type_id, depart="now", departLane=start_lane, departSpeed="max")

            wait_steps = int(2.0 / self.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.ego_vehicle_id in traci.vehicle.getIDList():
                    ego_appeared = True; self.ego_start_pos = traci.vehicle.getPosition(self.ego_vehicle_id); self.last_ego_pos = self.ego_start_pos
                    break
            if not ego_appeared: raise RuntimeError(f"Ego 车辆 '{self.ego_vehicle_id}' 在 {wait_steps} 步内未出现。")

        except traci.exceptions.TraCIException as e: print(f"错误：添加 Ego 车辆 '{self.ego_vehicle_id}' 失败：{e}"); raise RuntimeError("添加 Ego 车辆失败。")

    def reset(self) -> np.ndarray:
        """Reset environment for a new evaluation episode"""
        self._close()
        self._start_sumo()

        for _ in range(self.ego_insertion_delay_steps): # Delay loop
            try: traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                 print(f"延迟期间发生 TraCI 错误：{e}")
                 if "connection closed" in str(e).lower(): raise ConnectionError("SUMO 连接在延迟期间关闭。")

        self._add_ego_vehicle()
        self.current_step = 0
        self.collision_occurred = False # Ensure collision flag is reset on each reset
        self.last_raw_state = np.zeros(self.dqn_train_config.state_dim)
        # ego_start_pos and last_ego_pos are set in _add_ego_vehicle

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                raw_state = self._get_raw_state()
                self.last_raw_state = raw_state.copy()
            except traci.exceptions.TraCIException: print("警告：在 reset 中的初始状态获取期间发生 TraCI 异常。")
        else: print("警告：在 reset 中的 add/wait 后未立即找到 Ego 车辆。")

        return self.last_raw_state # Return raw state

    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[str, Tuple[float, float]]:
        """Get distance and relative speed of nearest vehicles (copied from dqn.py)"""
        max_dist = self.dqn_train_config.max_distance # Use distance from train config
        infos = { 'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0), 'left_back': (max_dist, 0.0), 'right_front': (max_dist, 0.0), 'right_back': (max_dist, 0.0) }
        veh_ids = traci.vehicle.getIDList()
        if ego_id not in veh_ids: return infos

        try: ego_road = traci.vehicle.getRoadID(ego_id);
        except traci.exceptions.TraCIException: return infos
        if not ego_road: return infos

        num_lanes_on_edge = self.num_lanes

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                if traci.vehicle.getRoadID(veh_id) != ego_road: continue
                veh_pos = traci.vehicle.getPosition(veh_id); veh_lane = traci.vehicle.getLaneIndex(veh_id); veh_speed = traci.vehicle.getSpeed(veh_id)
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
        return infos

    def _get_raw_state(self) -> np.ndarray:
        """Get current environment state (raw values before normalization, copied from dqn.py)"""
        state = np.zeros(self.dqn_train_config.state_dim, dtype=np.float32)
        ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList(): return self.last_raw_state

        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road: return self.last_raw_state

            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            self.last_ego_pos = ego_pos # Store current position
            num_lanes = self.num_lanes

            ego_lane = np.clip(ego_lane, 0, num_lanes - 1) # Clip lane index

            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (num_lanes - 1) else False

            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)

            state[0] = ego_speed; state[1] = float(ego_lane)
            state[2] = min(surround_info['front'][0], self.dqn_train_config.max_distance)
            state[3] = surround_info['front'][1]
            state[4] = min(surround_info['left_front'][0], self.dqn_train_config.max_distance)
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.dqn_train_config.max_distance)
            state[7] = min(surround_info['right_front'][0], self.dqn_train_config.max_distance)
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.dqn_train_config.max_distance)
            state[10] = 1.0 if can_change_left else 0.0
            state[11] = 1.0 if can_change_right else 0.0

            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"警告：在原始状态计算中检测到 NaN 或 Inf。使用最后有效的原始状态。")
                return self.last_raw_state

            self.last_raw_state = state.copy()

        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" not in str(e): print(f"警告：获取 {ego_id} 的原始状态时发生 TraCI 错误：{e}。返回最后已知的原始状态。")
            return self.last_raw_state
        except Exception as e: print(f"警告：获取 {ego_id} 的原始状态时发生未知错误：{e}。返回最后已知的原始状态。"); traceback.print_exc()
        return self.last_raw_state

        return state

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """Perform action, return (next_raw_state, done)"""
        done = False; ego_id = self.ego_vehicle_id

        if ego_id not in traci.vehicle.getIDList(): return self.last_raw_state, True

        try:
            current_lane = traci.vehicle.getLaneIndex(ego_id); num_lanes = self.num_lanes
            current_lane = np.clip(current_lane, 0, num_lanes - 1)

            if action == 1 and current_lane > 0: traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
            elif action == 2 and current_lane < (num_lanes - 1): traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)

            traci.simulationStep(); self.current_step += 1

            collisions = traci.simulation.getCollisions(); ego_collided_explicitly = False
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id:
                    self.collision_occurred = True; ego_collided_explicitly = True; done = True; break

            ego_exists_after_step = ego_id in traci.vehicle.getIDList()
            if not ego_exists_after_step and not ego_collided_explicitly: done = True

            if ego_exists_after_step: next_raw_state = self._get_raw_state()
            else: next_raw_state = self.last_raw_state

            if traci.simulation.getTime() >= 3600: done = True
            if self.current_step >= self.eval_config.EVAL_MAX_STEPS: done = True

        except traci.exceptions.TraCIException as e:
            is_not_known_error = "Vehicle '" + ego_id + "' is not known" in str(e)
            if not is_not_known_error: print(f"错误：在步骤 {self.current_step} 期间发生 TraCI 异常：{e}")
            if is_not_known_error and not self.collision_occurred: pass # Ego likely finished normally
            else: self.collision_occurred = True
            done = True; next_raw_state = self.last_raw_state
        except Exception as e:
            print(f"错误：在步骤 {self.current_step} 期间发生未知异常：{e}"); traceback.print_exc()
            done = True; self.collision_occurred = True; next_raw_state = self.last_raw_state

        return next_raw_state, done

    def get_vehicle_info(self):
        """Get current info like speed, lane, position, distance traveled"""
        ego_id = self.ego_vehicle_id
        if ego_id in traci.vehicle.getIDList():
            try:
                speed = traci.vehicle.getSpeed(ego_id); lane = traci.vehicle.getLaneIndex(ego_id)
                pos = traci.vehicle.getPosition(ego_id); self.last_ego_pos = pos
                dist_traveled = math.dist(pos, self.ego_start_pos) if self.ego_start_pos else 0.0
                return {"speed": speed, "lane": lane, "pos": pos, "dist": dist_traveled}
            except traci.exceptions.TraCIException:
                 dist_traveled = math.dist(self.last_ego_pos, self.ego_start_pos) if self.last_ego_pos and self.ego_start_pos else 0.0
                 return {"speed": 0, "lane": -1, "pos": self.last_ego_pos, "dist": dist_traveled}
        else:
            dist_traveled = math.dist(self.last_ego_pos, self.ego_start_pos) if self.last_ego_pos and self.ego_start_pos else 0.0
            return {"speed": 0, "lane": -1, "pos": self.last_ego_pos, "dist": dist_traveled}

    def _close(self):
        """Close SUMO instance"""
        if self.sumo_process:
            try: traci.close()
            except Exception: pass
            finally:
                try:
                    if self.sumo_process.poll() is None: self.sumo_process.terminate(); self.sumo_process.wait(timeout=2)
                except subprocess.TimeoutExpired: self.sumo_process.kill(); self.sumo_process.wait(timeout=1)
                except Exception as e: print(f"警告：SUMO 终止期间出错：{e}")
                self.sumo_process = None; self.traci_port = None; time.sleep(0.1)
        else: self.traci_port = None

#####################################
# 模型加载和动作选择                   #
#####################################

def load_model(model_path: str, model_type: str, train_config: Any, device: torch.device) -> nn.Module:
    """Load a trained model (DQN or PPO)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    # Load model based on type and respective config
    if model_type == 'dqn':
        # *** CORRECTED TYPE CHECK ***
        if not isinstance(train_config, DQN_Config_Base):
             raise TypeError(f"为 DQN 模型提供的配置不是 DQN_Config_Base 的实例, 而是 {type(train_config)}")
        model = QNetwork(train_config.state_dim, train_config.action_dim, train_config.hidden_size, train_config).to(device)
    elif model_type == 'ppo':
         # *** CORRECTED TYPE CHECK ***
        if not isinstance(train_config, PPO_Config_Base):
             raise TypeError(f"为 PPO 模型提供的配置不是 PPO_Config_Base 的实例, 而是 {type(train_config)}")
        model = PPO(train_config.state_dim, train_config.action_dim, train_config.hidden_size).to(device)
    else:
        raise ValueError(f"未知的模型类型：{model_type}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True if model_type=='dqn' else False)
        model.eval() # Set to evaluation mode
        print(f"成功从以下位置加载 {model_type.upper()} 模型：{os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"从 {model_path} 加载模型 state_dict 时出错：{e}")
        raise

def normalize_state(state_raw: np.ndarray, normalizer: Optional[RunningMeanStd], clip_val: float) -> np.ndarray:
    """Normalize state using the provided FROZEN normalizer instance"""
    if normalizer:
        mean = normalizer.mean; std = normalizer.std + 1e-8
        norm_state = (state_raw - mean) / std
        norm_state = np.clip(norm_state, -clip_val, clip_val)
        return norm_state.astype(np.float32)
    else:
        return state_raw.astype(np.float32)

def get_dqn_action(model: QNetwork, state_norm: np.ndarray, current_lane_idx: int, config: DQN_Config_Base, num_eval_lanes: int, device: torch.device) -> int:
    """Get action from DQN model (C51/Noisy aware)"""
    model.eval()
    if config.use_noisy_nets: model.reset_noise()

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        action_probs = model(state_tensor)
        support = torch.linspace(config.v_min, config.v_max, config.num_atoms).to(device)
        expected_q_values = (action_probs * support).sum(dim=2)

        q_values_masked = expected_q_values.clone()
        current_lane_idx = np.clip(current_lane_idx, 0, num_eval_lanes - 1)
        if current_lane_idx == 0: q_values_masked[0, 1] = -float('inf')
        if current_lane_idx >= num_eval_lanes - 1: q_values_masked[0, 2] = -float('inf')
        action = q_values_masked.argmax().item()
    return action

def get_ppo_action(model: PPO, state_norm: np.ndarray, current_lane_idx: int, num_eval_lanes: int, device: torch.device) -> int:
    """Get deterministic action from PPO actor"""
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        action_probs = model.get_action_probs(state_tensor)

        probs_masked = action_probs.clone()
        current_lane_idx = np.clip(current_lane_idx, 0, num_eval_lanes - 1)
        if current_lane_idx == 0: probs_masked[0, 1] = 0.0
        if current_lane_idx >= num_eval_lanes - 1: probs_masked[0, 2] = 0.0

        probs_sum = probs_masked.sum(dim=-1, keepdim=True)
        if probs_sum.item() > 1e-8: final_probs = probs_masked / probs_sum
        else: final_probs = torch.zeros_like(probs_masked); final_probs[0, 0] = 1.0
        action = final_probs.argmax().item()
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
    model_type: str, # 'dqn' or 'ppo'
    env: EvaluationEnv,
    train_config: Any, # DQN_Config_Base or PPO_Config_Base
    obs_normalizer: Optional[RunningMeanStd], # Initial (frozen) normalizer state
    device: torch.device,
    eval_config: EvalConfig
) -> Tuple[EpisodeResult, None]: # Return result, no need to return normalizer
    """Run a single evaluation episode for the given model"""

    state_raw = env.reset() # Returns raw state
    current_obs_normalizer = obs_normalizer # Use frozen normalizer

    done = False; step_count = 0; speeds = []; model_lane_changes = 0

    forced_attempts = 0; forced_agreed = 0; forced_executed_safe = 0; forced_executed_collision = 0
    monitoring_change = False; monitor_steps_left = 0
    monitor_target_action = -1; monitor_start_lane = -1; monitor_start_pos = None

    last_valid_vehicle_info = env.get_vehicle_info() # Store last valid info

    while not done and step_count < eval_config.EVAL_MAX_STEPS:
        clip_value = train_config.obs_norm_clip if hasattr(train_config, 'obs_norm_clip') else 5.0
        state_norm = normalize_state(state_raw, current_obs_normalizer, clip_value)

        if not np.any(np.isnan(state_raw)) and len(state_raw) > 1:
             current_lane_idx = int(round(state_raw[1]))
             current_lane_idx = np.clip(current_lane_idx, 0, eval_config.NUM_LANES - 1)
        else: current_lane_idx = last_valid_vehicle_info['lane'] if last_valid_vehicle_info['lane'] >= 0 else 0

        can_go_left = current_lane_idx > 0
        can_go_right = current_lane_idx < (eval_config.NUM_LANES - 1)
        target_action = -1

        # Check monitoring previous attempt
        if monitoring_change:
            monitor_steps_left -= 1
            current_vehicle_info = env.get_vehicle_info()
            current_pos = current_vehicle_info['pos']
            current_lane_after_step = current_vehicle_info['lane']

            lateral_dist = 0.0
            if current_pos and monitor_start_pos: lateral_dist = abs(current_pos[1] - monitor_start_pos[1])
            lane_changed_physically = (current_lane_after_step >= 0 and current_lane_after_step != monitor_start_lane)

            if lane_changed_physically or lateral_dist > eval_config.FORCE_CHANGE_SUCCESS_DIST:
                if env.collision_occurred: forced_executed_collision += 1
                else: forced_executed_safe += 1
                monitoring_change = False
            elif monitor_steps_left <= 0: monitoring_change = False
            elif env.collision_occurred: forced_executed_collision += 1; monitoring_change = False

        # Trigger new forced attempt if not monitoring
        if not monitoring_change and step_count > 0 and step_count % eval_config.FORCE_CHANGE_INTERVAL_STEPS == 0:
            if can_go_left and can_go_right: target_action = random.choice([1, 2])
            elif can_go_left: target_action = 1
            elif can_go_right: target_action = 2
            if target_action != -1: forced_attempts += 1

        # Get model action
        if model_type == 'dqn': action = get_dqn_action(model, state_norm, current_lane_idx, train_config, eval_config.NUM_LANES, device)
        elif model_type == 'ppo': action = get_ppo_action(model, state_norm, current_lane_idx, eval_config.NUM_LANES, device)
        else: action = 0

        # Handle forced change logic
        if target_action != -1:
            if action == target_action:
                forced_agreed += 1; monitoring_change = True
                monitor_steps_left = eval_config.FORCE_CHANGE_MONITOR_STEPS
                monitor_target_action = target_action; monitor_start_lane = current_lane_idx
                start_info = env.get_vehicle_info()
                monitor_start_pos = start_info['pos'] if start_info['lane'] != -1 else None
            else: pass # Continue with model's chosen action, don't monitor

        if action != 0 and not (monitoring_change and action == monitor_target_action): model_lane_changes += 1

        next_state_raw, done = env.step(action)

        state_raw = next_state_raw; step_count += 1
        vehicle_info = env.get_vehicle_info()
        if vehicle_info['lane'] != -1: speeds.append(vehicle_info['speed']); last_valid_vehicle_info = vehicle_info

        if env.collision_occurred:
            done = True
            if monitoring_change: forced_executed_collision += 1; monitoring_change = False

    # Episode End
    avg_speed = np.mean(speeds) if speeds else 0.0
    total_dist = last_valid_vehicle_info['dist']; collided = env.collision_occurred

    result = EpisodeResult(
        steps=step_count, collided=collided, avg_speed=avg_speed, total_dist=total_dist,
        forced_attempts=forced_attempts, forced_agreed=forced_agreed,
        forced_executed_safe=forced_executed_safe, forced_executed_collision=forced_executed_collision,
        model_lane_changes=model_lane_changes
    )
    return result, None


#####################################
# 主评估脚本                       #
#####################################
def main():
    eval_config = EvalConfig()
    # Need instances of the original training configs
    dqn_train_config = DQN_Config_Base()
    ppo_train_config = PPO_Config_Base()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evaluation_results_{timestamp}"; os.makedirs(output_dir, exist_ok=True)
    print(f"输出将保存在: {output_dir}")

    # --- File Checks ---
    if not os.path.exists(eval_config.DQN_MODEL_PATH): print(f"错误：未找到 DQN 模型路径：{eval_config.DQN_MODEL_PATH}"); sys.exit(1)
    if not os.path.exists(eval_config.PPO_MODEL_PATH): print(f"错误：未找到 PPO 模型路径：{eval_config.PPO_MODEL_PATH}"); sys.exit(1)
    if not os.path.exists(eval_config.EVAL_SUMO_CONFIG): print(f"错误：未找到评估 SUMO 配置：{eval_config.EVAL_SUMO_CONFIG}"); sys.exit(1)
    try: # Check referenced files
        with open(eval_config.EVAL_SUMO_CONFIG, 'r') as f: content = f.read()
        if 'net-file value="new.net.xml"' not in content: print("警告：'new.net.xml' 未在评估 sumocfg 中找到！")
        if 'route-files value="new.rou.xml"' not in content: print("警告：'new.rou.xml' 未在评估 sumocfg 中找到！")
        if not os.path.exists("new.net.xml"): print("警告：未找到文件 new.net.xml。")
        if not os.path.exists("new.rou.xml"): print("警告：未找到文件 new.rou.xml。")
    except Exception as e: print(f"警告：无法读取评估 SUMO 配置 {eval_config.EVAL_SUMO_CONFIG}: {e}")

    # --- Load Models ---
    print("\n--- 加载模型 ---")
    dqn_model = load_model(eval_config.DQN_MODEL_PATH, 'dqn', dqn_train_config, device)
    ppo_model = load_model(eval_config.PPO_MODEL_PATH, 'ppo', ppo_train_config, device)

    # --- Initialize Normalizers (FROZEN) ---
    dqn_obs_normalizer = RunningMeanStd(shape=(dqn_train_config.state_dim,)) if dqn_train_config.normalize_observations else None
    ppo_obs_normalizer = RunningMeanStd(shape=(ppo_train_config.state_dim,)) if ppo_train_config.normalize_observations else None
    if dqn_obs_normalizer or ppo_obs_normalizer:
        print("初始化（冻结的）归一化器以供评估使用。(注意：应加载训练统计数据)")
        # --- !!! ---
        # TODO: Add code here to load actual mean/std values from saved training checkpoints
        # Example (requires saving normalizer state during training):
        # if dqn_obs_normalizer and os.path.exists("dqn_normalizer_state.npz"):
        #    data = np.load("dqn_normalizer_state.npz")
        #    dqn_obs_normalizer.mean = data['mean']
        #    dqn_obs_normalizer.var = data['var']
        #    dqn_obs_normalizer.count = data['count'] # count is less critical if using EMA alpha
        #    print("加载了保存的 DQN 归一化器状态。")
        # if ppo_obs_normalizer and os.path.exists("ppo_normalizer_state.npz"):
        #    # ... load ppo normalizer state ...
        #    print("加载了保存的 PPO 归一化器状态。")
        # --- !!! ---


    # --- Initialize Environment ---
    base_seed = eval_config.EVAL_SEED
    env = EvaluationEnv(eval_config, dqn_train_config, base_seed) # Pass DQN config for state dim etc.

    # --- Run Evaluation ---
    print(f"\n--- 运行评估 ({eval_config.EVAL_EPISODES} 个回合/模型) ---")
    dqn_results_list: List[EpisodeResult] = []
    ppo_results_list: List[EpisodeResult] = []

    for i in tqdm(range(eval_config.EVAL_EPISODES), desc="评估回合"):
        episode_seed = base_seed + i

        # --- Evaluate DQN ---
        env.sumo_seed = episode_seed
        try:
            dqn_result, _ = evaluate_episode(dqn_model, 'dqn', env, dqn_train_config, dqn_obs_normalizer, device, eval_config)
            dqn_results_list.append(dqn_result)
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e: print(f"\nDQN 评估回合 {i+1} 期间出错：{e}"); time.sleep(0.5); env._close(); time.sleep(1); continue # Skip to next episode on error
        except Exception as e_other: print(f"\nDQN 评估回合 {i+1} 期间发生意外错误：{e_other}"); traceback.print_exc(); time.sleep(0.5); env._close(); time.sleep(1); continue

        # --- Evaluate PPO ---
        env.sumo_seed = episode_seed
        try:
            ppo_result, _ = evaluate_episode(ppo_model, 'ppo', env, ppo_train_config, ppo_obs_normalizer, device, eval_config)
            ppo_results_list.append(ppo_result)
        except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e: print(f"\nPPO 评估回合 {i+1} 期间出错：{e}"); time.sleep(0.5); env._close(); time.sleep(1); continue
        except Exception as e_other: print(f"\nPPO 评估回合 {i+1} 期间发生意外错误：{e_other}"); traceback.print_exc(); time.sleep(0.5); env._close(); time.sleep(1); continue

    env._close(); print("\n--- 评估完成 ---")

    # --- Aggregate and Compare Results ---
    if not dqn_results_list or not ppo_results_list: print("未收集到足够的评估结果。正在退出。"); return

    results = {'dqn': {}, 'ppo': {}}
    metrics_definitions = [ # (key, name, format, higher_is_better, std_dev_key)
        ('avg_steps', '平均步数', '.1f', True, 'std_steps'),
        ('std_steps', '步数标准差', '.1f', False, None),
        ('avg_speed', '平均速度 (米/秒)', '.2f', True, 'std_speed'),
        ('std_speed', '速度标准差 (米/秒)', '.2f', False, None),
        ('avg_dist', '平均距离 (米)', '.1f', True, None),
        ('collision_rate', '碰撞率 (%)', '.1f', False, None),
        ('avg_model_lc', '模型发起的平均换道次数', '.1f', None, None),
        ('total_forced_attempts', '强制换道尝试总数', 'd', None, None),
        ('forced_agreement_rate', '强制换道同意率 (%)', '.1f', True, None),
        ('forced_execution_success_rate', '强制换道执行成功率 (% 同意)', '.1f', True, None),
        ('forced_execution_collision_rate', '强制换道执行碰撞率 (% 同意)', '.1f', False, None),
    ]

    for model_key, results_list in [('dqn', dqn_results_list), ('ppo', ppo_results_list)]:
        total_episodes = len(results_list); results[model_key]['total_episodes'] = total_episodes
        if total_episodes == 0: continue
        results[model_key]['avg_steps'] = np.mean([r.steps for r in results_list]); results[model_key]['std_steps'] = np.std([r.steps for r in results_list])
        results[model_key]['collision_rate'] = np.mean([r.collided for r in results_list]) * 100
        results[model_key]['avg_speed'] = np.mean([r.avg_speed for r in results_list]); results[model_key]['std_speed'] = np.std([r.avg_speed for r in results_list])
        results[model_key]['avg_dist'] = np.mean([r.total_dist for r in results_list]); results[model_key]['avg_model_lc'] = np.mean([r.model_lane_changes for r in results_list])
        total_forced_attempts = sum(r.forced_attempts for r in results_list); total_forced_agreed = sum(r.forced_agreed for r in results_list)
        total_forced_executed_safe = sum(r.forced_executed_safe for r in results_list); total_forced_executed_collision = sum(r.forced_executed_collision for r in results_list)
        results[model_key]['total_forced_attempts'] = total_forced_attempts
        results[model_key]['forced_agreement_rate'] = (total_forced_agreed / total_forced_attempts * 100) if total_forced_attempts > 0 else 0
        results[model_key]['forced_execution_success_rate'] = (total_forced_executed_safe / total_forced_agreed * 100) if total_forced_agreed > 0 else 0
        results[model_key]['forced_execution_collision_rate'] = (total_forced_executed_collision / total_forced_agreed * 100) if total_forced_agreed > 0 else 0

    # --- Print and Save Text Comparison ---
    print("\n--- 结果比较 (文本) ---")
    comparison_lines = []; header1 = f"{'指标':<35} | {'DQN':<20} | {'PPO':<20}"; header2 = "-" * (35 + 20 + 20 + 5)
    comparison_lines.append(header1); comparison_lines.append(header2); print(header1); print(header2)
    for key, name, fmt, _, _ in metrics_definitions:
        dqn_val = results.get('dqn', {}).get(key, 'N/A'); ppo_val = results.get('ppo', {}).get(key, 'N/A')
        dqn_str = format(dqn_val, fmt) if isinstance(dqn_val, (int, float)) else str(dqn_val)
        ppo_str = format(ppo_val, fmt) if isinstance(ppo_val, (int, float)) else str(ppo_val)
        line = f"{name:<35} | {dqn_str:<20} | {ppo_str:<20}"; comparison_lines.append(line); print(line)
    comparison_lines.append(header2); print(header2)
    text_results_filename = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    try:
        with open(text_results_filename, 'w', encoding='utf-8') as f:
            f.write(f"评估运行时间: {timestamp}\n"); f.write(f"评估回合数: {eval_config.EVAL_EPISODES}\n"); f.write(f"最大步数/回合: {eval_config.EVAL_MAX_STEPS}\n")
            f.write(f"DQN 模型: {os.path.basename(eval_config.DQN_MODEL_PATH)}\n"); f.write(f"PPO 模型: {os.path.basename(eval_config.PPO_MODEL_PATH)}\n")
            f.write(f"SUMO 配置: {eval_config.EVAL_SUMO_CONFIG}\n"); f.write(f"Ego 插入延迟: {eval_config.EGO_INSERTION_DELAY_SECONDS} 秒\n")
            f.write("\n--- 结果比较 ---\n"); f.write("\n".join(comparison_lines))
        print(f"文本结果摘要已保存至: {text_results_filename}")
    except Exception as e: print(f"保存文本结果时出错：{e}")

    # --- Generate Separate Comparison Plots ---
    print("\n--- 生成单独的对比图表 ---")
    models = ['DQN', 'PPO']; colors = ['deepskyblue', 'lightcoral']
    plot_metrics = [ # (key, title, ylabel, filename_suffix, std_dev_key)
        ('avg_steps', '每回合平均步数', '步数', 'avg_steps', 'std_steps'), ('avg_speed', '平均速度', '速度 (米/秒)', 'avg_speed', 'std_speed'),
        ('collision_rate', '总碰撞率', '比率 (%)', 'collision_rate', None), ('avg_dist', '平均行驶距离', '距离 (米)', 'avg_dist', None),
        ('avg_model_lc', '模型发起的平均换道次数', '次数', 'avg_model_lc', None), ('forced_agreement_rate', '强制换道: 模型同意率', '比率 (%)', 'forced_agreement_rate', None),
        ('forced_execution_success_rate', '强制换道: 执行成功率\n(同意换道的百分比)', '比率 (%)', 'forced_exec_success_rate', None),
        ('forced_execution_collision_rate', '强制换道: 执行碰撞率\n(同意换道的百分比)', '比率 (%)', 'forced_exec_collision_rate', None),]

    for key, title, ylabel, fname_suffix, std_key in plot_metrics:
        plt.figure(figsize=(6, 5)); plt.title(f'{title} (DQN vs PPO)', fontsize=14)
        values = [results.get('dqn', {}).get(key, 0), results.get('ppo', {}).get(key, 0)]
        std_devs = [0, 0]; has_std = False
        if std_key:
             std_dqn = results.get('dqn', {}).get(std_key, 0); std_ppo = results.get('ppo', {}).get(std_key, 0)
             std_devs = [std_dqn, std_ppo]; has_std = True
        bars = plt.bar(models, values, color=colors, yerr=std_devs if has_std else None, capsize=5 if has_std else 0)
        for i, bar in enumerate(bars):
             yval = bar.get_height(); y_offset = yval + (std_devs[i] * 0.5 if has_std else 0) + max(values)*0.02
             label_text = f"{yval:.1f}%" if '%' in ylabel else (f"{yval:.2f}" if isinstance(yval, float) else f"{yval}")
             plt.text(bar.get_x() + bar.get_width()/2.0, y_offset, label_text, ha='center', va='bottom')
        plt.ylabel(ylabel)
        max_val_display = max(yval + std_devs[i] if has_std else yval for i, yval in enumerate(values)); min_val_display = min(yval - std_devs[i] if has_std else yval for i, yval in enumerate(values))
        if '%' in ylabel: plt.ylim(0, max(1, max_val_display * 1.1, 10)); plt.ylim(0, 105) if max_val_display > 90 else None
        elif max_val_display > 0 : plt.ylim(0, max_val_display * 1.2)
        elif min_val_display < 0: plt.ylim(min_val_display * 1.2, max(1, max_val_display * 1.2) )
        else: plt.ylim(0, max(1, max_val_display*1.2))
        plt.grid(axis='y', linestyle='--'); plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"plot_{fname_suffix}_{timestamp}.png")
        plt.savefig(plot_filename); print(f"对比图表已保存至: {plot_filename}"); plt.close()

    # --- Save JSON Data ---
    data_filename = os.path.join(output_dir, f"evaluation_data_{timestamp}.json")
    results['dqn']['raw_results'] = [r._asdict() for r in dqn_results_list]
    results['ppo']['raw_results'] = [r._asdict() for r in ppo_results_list]
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        print(f"比较数据已保存至: {data_filename}")
    except Exception as e: print(f"保存比较数据时出错：{e}")

if __name__ == "__main__":
    main()