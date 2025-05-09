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
import math
import copy  # For deep copying configs/normalizers

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Or 'Microsoft YaHei' etc.
plt.rcParams['axes.unicode_minus'] = False

# --- 导入必要的组件 ---
# We'll use components from dqnyh.py as the primary source for network definitions
# as both training scripts are DQN-based and dqnyh.py is marked as "V2 Optimized".
# We'll import Config from both to correctly initialize models if their specific params differ.
try:
    from dqnyh import Config as DQNYH_Config, QNetwork, NoisyLinear, RunningMeanStd
    from dqnnew import Config as DQNNEW_Config
except ImportError as e:
    print(f"Error importing from dqnyh.py or dqnnew.py: {e}")
    print("Please ensure dqnyh.py and dqnnew.py are in the same directory or accessible in the Python path.")
    sys.exit(1)
except AttributeError as e_attr:
    print(f"AttributeError during import (maybe a class name changed?): {e_attr}")
    sys.exit(1)


#####################################
# 评估配置                         #
#####################################
class EvalConfig:
    # --- 模型路径 (请修改为您的实际模型文件路径) ---
    MODEL_A_NAME = "DQN_Optimized_V2"  # Model from dqnyh.py
    MODEL_A_PATH = "model_from_dqnyh.pth"  # 修改为您的 DQN (dqnyh.py) 模型路径
    MODEL_A_TRAIN_SCRIPT_TYPE = "dqnyh"  # Internal type to load correct config

    MODEL_B_NAME = "DQN_Revised"  # Model from dqnnew.py
    MODEL_B_PATH = "model_from_dqnnew.pth"  # 修改为您的 DQN (dqnnew.py) 模型路径
    MODEL_B_TRAIN_SCRIPT_TYPE = "dqnnew"  # Internal type to load correct config

    # --- SUMO 配置 ---
    EVAL_SUMO_BINARY = "sumo"  # 使用 "sumo-gui" 进行可视化, 或 "sumo"
    EVAL_SUMO_CONFIG = "new.sumocfg"  # 使用您提供的评估配置文件
    EVAL_STEP_LENGTH = 0.2
    EVAL_PORT_RANGE = (8910, 8920)  # 评估使用的不同端口范围

    # --- 评估参数 ---
    EVAL_EPISODES = 500  # 每个模型的评估回合数 (可调整)
    EVAL_MAX_STEPS = 1500  # 每回合最大步数 (例如 300 秒)
    EVAL_SEED = 42  # SUMO 启动种子 (用于可复现性)
    NUM_LANES = 4  # new.net.xml 中的车道数 (0, 1, 2, 3)
    EGO_INSERTION_DELAY_SECONDS = 50.0  # Ego车辆插入延迟

    # --- 强制换道尝试逻辑 (与 pinggunew.py 保持一致) ---
    FORCE_CHANGE_INTERVAL_STEPS = 50  # 每 X 步提出一次强制换道
    FORCE_CHANGE_MONITOR_STEPS = 15  # 等待/监控 *同意的* 换道完成的步数
    FORCE_CHANGE_SUCCESS_DIST = 1.0  # 认为换道成功的最小横向距离

    # --- Ego 车辆 ID 和类型 ---
    EVAL_EGO_ID = "eval_ego_compare"  # 评估中使用的 Ego ID
    EVAL_EGO_TYPE = "car_ego_eval_compare"  # 评估中使用的 Ego 类型


#####################################
# 辅助函数 (复制/改编自 pinggunew.py) #
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
        if os.name == 'nt':  # Windows
            result1 = os.system("taskkill /f /im sumo.exe >nul 2>&1")
            result2 = os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
            killed = (result1 == 0 or result2 == 0)
        else:  # Linux/macOS
            result1 = os.system("pkill -f sumo > /dev/null 2>&1")
            result2 = os.system("pkill -f sumo-gui > /dev/null 2>&1")
            killed = (result1 == 0 or result2 == 0)
    except Exception as e:
        print(f"Warning: Error killing SUMO processes: {e}")
    time.sleep(0.1)


def calculate_rolling_average(data, window):
    if len(data) < window:
        return np.array([])
    data_np = np.array(data, dtype=float)
    if np.any(np.isnan(data_np)):
        # print(f"Warning: NaNs detected in rolling average input data. Replacing with mean.") # Debug
        data_np = np.nan_to_num(data_np, nan=np.nanmean(data_np))

    if len(data_np) < window:
        return np.array([])
    return np.convolve(data_np, np.ones(window) / window, mode='valid')


#####################################
# 评估环境封装                       #
#####################################
class EvaluationEnv:
    def __init__(self, eval_config_main: EvalConfig, base_train_config: Any, sumo_seed: int):
        self.eval_config = eval_config_main
        # base_train_config is used for generic parameters like state_dim, max_distance
        # Assuming these are consistent across the models being compared.
        self.base_train_config = base_train_config
        self.sumo_binary = self.eval_config.EVAL_SUMO_BINARY
        self.config_path = self.eval_config.EVAL_SUMO_CONFIG
        self.step_length = self.eval_config.EVAL_STEP_LENGTH
        self.ego_vehicle_id = self.eval_config.EVAL_EGO_ID
        self.ego_type_id = self.eval_config.EVAL_EGO_TYPE
        self.port_range = self.eval_config.EVAL_PORT_RANGE
        self.num_lanes = self.eval_config.NUM_LANES  # From eval_config
        self.sumo_seed = sumo_seed
        self.ego_insertion_delay_steps = int(self.eval_config.EGO_INSERTION_DELAY_SECONDS / self.step_length)

        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_port: Optional[int] = None
        self.last_raw_state = np.zeros(self.base_train_config.state_dim)
        self.current_step = 0
        self.collision_occurred = False
        self.ego_start_pos = None
        self.ego_route_id = "route_E0"  # Matches new.rou.xml
        self.last_ego_pos = None

    def _start_sumo(self):
        kill_sumo_processes()
        try:
            self.traci_port = get_available_port(self.port_range[0], self.port_range[1])
        except IOError as e:
            print(f"Error: Could not find available port: {e}");
            sys.exit(1)

        sumo_cmd = [
            self.sumo_binary, "-c", self.config_path,
            "--remote-port", str(self.traci_port), "--step-length", str(self.step_length),
            "--collision.check-junctions", "true", "--collision.action", "warn",
            "--time-to-teleport", "-1", "--no-warnings", "true", "--seed", str(self.sumo_seed)
        ]
        if self.sumo_binary == "sumo-gui":
            sumo_cmd.extend(["--quit-on-end", "true", "--start"])  # Add --start for GUI

        try:
            stdout_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
            stderr_target = subprocess.DEVNULL if self.sumo_binary == "sumo" else None
            self.sumo_process = subprocess.Popen(sumo_cmd, stdout=stdout_target, stderr=stderr_target)
        except FileNotFoundError:
            print(f"Error: SUMO executable '{self.sumo_binary}' not found."); sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to start SUMO process: {e}"); sys.exit(1)

        connection_attempts = 10  # Increased attempts for GUI
        for attempt in range(connection_attempts):
            try:
                time.sleep(1.5 + attempt * 0.5)  # Increased sleep for GUI
                traci.init(self.traci_port)
                return
            except traci.exceptions.TraCIException as te:
                if attempt == connection_attempts - 1:
                    print(f"Max TraCI connection attempts reached ({te}).");
                    self._close();
                    raise ConnectionError(f"Could not connect to SUMO (Port: {self.traci_port}).")
            except Exception as e:
                print(f"Unexpected error connecting TraCI: {e}"); self._close(); raise ConnectionError(
                    f"Unknown error connecting to SUMO (Port: {self.traci_port}).")

    def _add_ego_vehicle(self):
        if self.ego_route_id not in traci.route.getIDList():
            edge_list = list(traci.edge.getIDList())
            first_edge = "E0" if "E0" in edge_list else (edge_list[0] if edge_list else None)
            if first_edge:
                print(f"Warning: Route '{self.ego_route_id}' not found. Creating route from edge '{first_edge}'.")
                try:
                    traci.route.add(self.ego_route_id, [first_edge])
                except traci.exceptions.TraCIException as e_add:
                    raise RuntimeError(f"Failed to add route '{self.ego_route_id}': {e_add}")
            else:
                raise RuntimeError(f"Route '{self.ego_route_id}' not found and no suitable edge available.")

        if self.ego_type_id not in traci.vehicletype.getIDList():
            base_type = "car"  # From new.rou.xml
            if base_type not in traci.vehicletype.getIDList():
                available_types = traci.vehicletype.getIDList()
                base_type = available_types[0] if available_types else None
                if not base_type: raise RuntimeError("No vehicle types found in simulation.")
                print(f"Warning: Base type 'car' not found. Using fallback '{base_type}'.")
            try:
                traci.vehicletype.copy(base_type, self.ego_type_id)
                # Set a distinct color for this evaluation ego, e.g., Green
                traci.vehicletype.setColor(self.ego_type_id, (0, 150, 0, 255))  # R,G,B,A
                traci.vehicletype.setParameter(self.ego_type_id, "lcStrategic", "1.0")  # Basic lane change params
                traci.vehicletype.setParameter(self.ego_type_id, "lcSpeedGain", "1.0")
                traci.vehicletype.setParameter(self.ego_type_id, "lcCooperative", "0.5")
            except traci.exceptions.TraCIException as e:
                print(f"Warning: Failed to set parameters for Ego type '{self.ego_type_id}': {e}")

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                traci.vehicle.remove(self.ego_vehicle_id); time.sleep(0.1)
            except traci.exceptions.TraCIException as e:
                print(f"Warning: Failed to remove lingering Ego '{self.ego_vehicle_id}': {e}")

        try:
            start_lane = random.choice(range(self.num_lanes))
            traci.vehicle.add(vehID=self.ego_vehicle_id, routeID=self.ego_route_id, typeID=self.ego_type_id,
                              depart="now", departLane=start_lane, departSpeed="max")
            wait_steps = int(2.0 / self.step_length)
            ego_appeared = False
            for _ in range(wait_steps):
                traci.simulationStep()
                if self.ego_vehicle_id in traci.vehicle.getIDList():
                    ego_appeared = True;
                    self.ego_start_pos = traci.vehicle.getPosition(self.ego_vehicle_id);
                    self.last_ego_pos = self.ego_start_pos;
                    break
            if not ego_appeared: raise RuntimeError(f"Ego vehicle '{self.ego_vehicle_id}' did not appear after add.")
            # Explicitly set color again if needed, though type color should apply
            # traci.vehicle.setColor(self.ego_vehicle_id, (0, 150, 0, 255))
        except traci.exceptions.TraCIException as e:
            raise RuntimeError(f"Failed adding Ego vehicle '{self.ego_vehicle_id}': {e}")

    def reset(self) -> np.ndarray:
        self._close()
        self._start_sumo()
        # print(f"Running initial {self.ego_insertion_delay_steps} steps for traffic buildup...") # Debug
        for _ in range(self.ego_insertion_delay_steps):
            try:
                traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                raise ConnectionError(f"SUMO connection closed during delay: {e}")
        # print("Adding Ego vehicle...") # Debug
        self._add_ego_vehicle()
        self.current_step = 0;
        self.collision_occurred = False
        self.last_raw_state = np.zeros(self.base_train_config.state_dim)

        if self.ego_vehicle_id in traci.vehicle.getIDList():
            try:
                for _ in range(int(1.0 / self.step_length)): traci.simulationStep()  # Stabilize
                if self.ego_vehicle_id in traci.vehicle.getIDList():
                    self.last_raw_state = self._get_raw_state().copy()
                    if np.any(np.isnan(self.last_raw_state)): self.last_raw_state = np.nan_to_num(self.last_raw_state)
            except traci.exceptions.TraCIException as e:
                print(f"Warning: TraCI exception during initial state fetch: {e}.")
        else:
            print("Warning: Ego not found after add/stabilize in reset.")
        return self.last_raw_state

    def _get_surrounding_vehicle_info(self, ego_id: str, ego_speed: float, ego_pos: tuple, ego_lane: int) -> Dict[
        str, Tuple[float, float]]:
        max_dist = self.base_train_config.max_distance
        infos = {'front': (max_dist, 0.0), 'left_front': (max_dist, 0.0), 'left_back': (max_dist, 0.0),
                 'right_front': (max_dist, 0.0), 'right_back': (max_dist, 0.0)}
        try:
            veh_ids = traci.vehicle.getIDList()
            if ego_id not in veh_ids: return infos
            ego_road = traci.vehicle.getRoadID(ego_id);
            if not ego_road: return infos
        except traci.exceptions.TraCIException:
            return infos

        for veh_id in veh_ids:
            if veh_id == ego_id: continue
            try:
                if traci.vehicle.getRoadID(veh_id) != ego_road: continue
                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                if not (0 <= veh_lane < self.num_lanes): continue
                dx = veh_pos[0] - ego_pos[0];
                longitudinal_distance = abs(dx)
                if longitudinal_distance >= max_dist: continue
                rel_speed = ego_speed - veh_speed
                if veh_lane == ego_lane:
                    if dx > 0 and longitudinal_distance < infos['front'][0]: infos['front'] = (longitudinal_distance,
                                                                                               rel_speed)
                elif veh_lane == ego_lane - 1:
                    if dx > 0 and longitudinal_distance < infos['left_front'][0]:
                        infos['left_front'] = (longitudinal_distance, rel_speed)
                    elif dx <= 0 and longitudinal_distance < infos['left_back'][0]:
                        infos['left_back'] = (longitudinal_distance, rel_speed)
                elif veh_lane == ego_lane + 1:
                    if dx > 0 and longitudinal_distance < infos['right_front'][0]:
                        infos['right_front'] = (longitudinal_distance, rel_speed)
                    elif dx <= 0 and longitudinal_distance < infos['right_back'][0]:
                        infos['right_back'] = (longitudinal_distance, rel_speed)
            except traci.exceptions.TraCIException:
                continue
        return infos

    def _get_raw_state(self) -> np.ndarray:
        state = np.zeros(self.base_train_config.state_dim, dtype=np.float32)
        ego_id = self.ego_vehicle_id
        if ego_id not in traci.vehicle.getIDList(): return self.last_raw_state
        try:
            current_road = traci.vehicle.getRoadID(ego_id)
            if not current_road: return self.last_raw_state
            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_lane = traci.vehicle.getLaneIndex(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            self.last_ego_pos = ego_pos
            ego_lane = np.clip(ego_lane, 0, self.num_lanes - 1)
            can_change_left = traci.vehicle.couldChangeLane(ego_id, -1) if ego_lane > 0 else False
            can_change_right = traci.vehicle.couldChangeLane(ego_id, 1) if ego_lane < (self.num_lanes - 1) else False
            surround_info = self._get_surrounding_vehicle_info(ego_id, ego_speed, ego_pos, ego_lane)
            state[0] = ego_speed;
            state[1] = float(ego_lane)
            state[2] = min(surround_info['front'][0], self.base_train_config.max_distance);
            state[3] = surround_info['front'][1]
            state[4] = min(surround_info['left_front'][0], self.base_train_config.max_distance);
            state[5] = surround_info['left_front'][1]
            state[6] = min(surround_info['left_back'][0], self.base_train_config.max_distance)
            state[7] = min(surround_info['right_front'][0], self.base_train_config.max_distance);
            state[8] = surround_info['right_front'][1]
            state[9] = min(surround_info['right_back'][0], self.base_train_config.max_distance)
            state[10] = 1.0 if can_change_left else 0.0;
            state[11] = 1.0 if can_change_right else 0.0
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                state = np.nan_to_num(state, nan=0.0, posinf=self.base_train_config.max_distance,
                                      neginf=-self.base_train_config.max_distance)
                if np.any(np.isfinite(self.last_raw_state)): return self.last_raw_state
            self.last_raw_state = state.copy()
        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" not in str(e): print(
                f"Warning: TraCI Error _get_raw_state: {e}.")
            return self.last_raw_state
        except Exception as e:
            print(f"Warning: Unknown error _get_raw_state: {e}"); traceback.print_exc(); return self.last_raw_state
        return self.last_raw_state

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        done = False;
        ego_id = self.ego_vehicle_id
        if ego_id not in traci.vehicle.getIDList(): self.collision_occurred = True; return self.last_raw_state, True
        try:
            current_lane = np.clip(traci.vehicle.getLaneIndex(ego_id), 0, self.num_lanes - 1)
            if action == 1 and current_lane > 0:
                traci.vehicle.changeLane(ego_id, current_lane - 1, duration=1.0)
            elif action == 2 and current_lane < (self.num_lanes - 1):
                traci.vehicle.changeLane(ego_id, current_lane + 1, duration=1.0)
            traci.simulationStep();
            self.current_step += 1
            collisions = traci.simulation.getCollisions()
            for col in collisions:
                if col.collider == ego_id or col.victim == ego_id: self.collision_occurred = True; done = True; break
            ego_exists_after_step = ego_id in traci.vehicle.getIDList()
            if not ego_exists_after_step and not self.collision_occurred: done = True
            next_raw_state = self._get_raw_state() if ego_exists_after_step else self.last_raw_state
            if traci.simulation.getTime() >= 3600 or self.current_step >= self.eval_config.EVAL_MAX_STEPS: done = True
        except traci.exceptions.TraCIException as e:
            if "Vehicle '" + ego_id + "' is not known" not in str(e): print(
                f"Error: TraCI Exception step {self.current_step}: {e}")
            if "Vehicle '" + ego_id + "' is not known" in str(e) and not self.collision_occurred:
                pass
            else:
                self.collision_occurred = True
            done = True;
            next_raw_state = self.last_raw_state
        except Exception as e:
            print(
                f"Error: Unknown Exception step {self.current_step}: {e}"); traceback.print_exc(); done = True; self.collision_occurred = True; next_raw_state = self.last_raw_state
        return next_raw_state, done

    def get_vehicle_info(self):
        ego_id = self.ego_vehicle_id
        if ego_id in traci.vehicle.getIDList():
            try:
                speed = traci.vehicle.getSpeed(ego_id);
                lane = traci.vehicle.getLaneIndex(ego_id);
                pos = traci.vehicle.getPosition(ego_id)
                self.last_ego_pos = pos
                dist_traveled = math.dist(pos, self.ego_start_pos) if self.ego_start_pos else 0.0
                front_dist = self.last_raw_state[2] if len(self.last_raw_state) > 2 and np.isfinite(
                    self.last_raw_state[2]) else self.base_train_config.max_distance
                return {"speed": speed, "lane": lane, "pos": pos, "dist": dist_traveled,
                        "front_dist": max(0, front_dist)}
            except traci.exceptions.TraCIException:
                pass  # Fall through to return based on last known
        dist_traveled = math.dist(self.last_ego_pos,
                                  self.ego_start_pos) if self.last_ego_pos and self.ego_start_pos else 0.0
        front_dist = self.last_raw_state[2] if len(self.last_raw_state) > 2 and np.isfinite(
            self.last_raw_state[2]) else self.base_train_config.max_distance
        return {"speed": 0, "lane": -1, "pos": self.last_ego_pos, "dist": dist_traveled,
                "front_dist": max(0, front_dist)}

    def _close(self):
        if self.sumo_process:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                try:
                    if self.sumo_process.poll() is None: self.sumo_process.terminate(); self.sumo_process.wait(
                        timeout=2)
                except subprocess.TimeoutExpired:
                    self.sumo_process.kill(); self.sumo_process.wait(timeout=1)
                except Exception as e:
                    print(f"Warning: Error during SUMO termination: {e}")
                self.sumo_process = None;
                self.traci_port = None;
                time.sleep(0.1)
        else:
            self.traci_port = None


#####################################
# 模型加载和动作选择                   #
#####################################
def load_dqn_model(model_path: str, specific_train_config: Any, device: torch.device) -> QNetwork:
    """Loads a trained DQN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Instantiate QNetwork using the definitions from dqnyh.py (imported as QNetwork, NoisyLinear)
    # Pass the specific_train_config (either DQNYH_Config or DQNNEW_Config instance)
    # This config object contains num_atoms, v_min, v_max, use_noisy_nets etc.
    model = QNetwork(specific_train_config.state_dim,
                     specific_train_config.action_dim,
                     specific_train_config.hidden_size,
                     specific_train_config).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        print(f"Successfully loaded DQN model from: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"Error loading DQN model state_dict from {model_path}: {e}")
        raise


def normalize_state(state_raw: np.ndarray, normalizer: Optional[RunningMeanStd], clip_val: float) -> np.ndarray:
    """Normalize state using the provided FROZEN normalizer instance"""
    if normalizer:
        mean = normalizer.mean;
        std = normalizer.std + 1e-8
        if state_raw.ndim == 1 and mean.ndim == 1 and state_raw.shape[0] == mean.shape[0]:
            norm_state = (state_raw - mean) / std
        elif state_raw.ndim == 2 and mean.ndim == 1 and state_raw.shape[1] == mean.shape[0]:
            norm_state = (state_raw - mean[np.newaxis, :]) / std[np.newaxis, :]
        else:
            print(
                f"Warning: Shape mismatch in normalize_state. Raw: {state_raw.shape}, Mean: {mean.shape}. Returning raw.");
            return state_raw.astype(np.float32)
        norm_state = np.clip(norm_state, -clip_val, clip_val)
        return norm_state.astype(np.float32)
    return state_raw.astype(np.float32)


def get_dqn_model_action(model: QNetwork, state_norm: np.ndarray, current_lane_idx: int,
                         train_config: Any,  # DQNYH_Config or DQNNEW_Config instance
                         num_eval_lanes: int, device: torch.device) -> int:
    """Get action from a DQN model (C51/Noisy aware)"""
    model.eval()
    if train_config.use_noisy_nets: model.reset_noise()

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        action_probs = model(state_tensor)
        support = torch.linspace(train_config.v_min, train_config.v_max, train_config.num_atoms).to(device)
        expected_q_values = (action_probs * support).sum(dim=2)
        q_values_masked = expected_q_values.clone()
        current_lane_idx = np.clip(current_lane_idx, 0, num_eval_lanes - 1)
        if current_lane_idx == 0: q_values_masked[0, 1] = -float('inf')
        if current_lane_idx >= num_eval_lanes - 1: q_values_masked[0, 2] = -float('inf')
        action = q_values_masked.argmax().item()
    return action


#####################################
# 评估回合运行器                    #
#####################################
EpisodeResult = namedtuple('EpisodeResult', [
    'steps', 'collided', 'avg_speed', 'total_dist',
    'forced_attempts', 'forced_agreed', 'forced_disagreed',
    'forced_executed_safe', 'forced_executed_collision',
    'agreed_lane_changes', 'min_front_dist'
])
EvaluateEpisodeReturnType = Tuple[EpisodeResult, List[float], List[int], List[int], List[float]]


def evaluate_episode(
        model: QNetwork,  # Both models are QNetwork
        train_config: Any,  # DQNYH_Config or DQNNEW_Config instance
        env: EvaluationEnv,
        obs_normalizer: Optional[RunningMeanStd],
        device: torch.device,
        eval_config_main: EvalConfig  # Main EvalConfig
) -> EvaluateEpisodeReturnType:
    """Run a single evaluation episode with forced change logic."""
    state_raw = env.reset()
    if np.any(np.isnan(state_raw)):
        print("Error: NaN in initial state. Aborting episode.")
        dummy_result = EpisodeResult(0, True, 0, 0, 0, 0, 0, 0, 0, 0, env.base_train_config.max_distance)
        return dummy_result, [], [], [], []

    done = False;
    step_count = 0;
    agreed_lane_changes = 0
    min_front_dist_episode = env.base_train_config.max_distance
    all_speeds_ep, all_lanes_ep, all_actions_ep, all_front_dists_ep = [], [], [], []
    forced_attempts, forced_agreed, forced_disagreed = 0, 0, 0
    forced_executed_safe, forced_executed_collision = 0, 0
    monitoring_change, monitor_steps_left, monitor_target_action = False, 0, -1
    monitor_start_lane, monitor_start_pos = -1, None
    last_valid_vehicle_info = env.get_vehicle_info()

    while not done and step_count < eval_config_main.EVAL_MAX_STEPS:
        clip_value = train_config.obs_norm_clip if hasattr(train_config, 'obs_norm_clip') else 5.0
        state_norm = normalize_state(state_raw, obs_normalizer, clip_value)
        if np.any(np.isnan(state_norm)): print(
            f"Error: NaN in normalized state step {step_count}. Aborting."); done = True; env.collision_occurred = True; break

        current_lane_idx = int(round(state_raw[1])) if not np.any(np.isnan(state_raw)) and len(state_raw) > 1 else (
            last_valid_vehicle_info['lane'] if last_valid_vehicle_info['lane'] >= 0 else 0)
        current_lane_idx = np.clip(current_lane_idx, 0, eval_config_main.NUM_LANES - 1)
        can_go_left = current_lane_idx > 0
        can_go_right = current_lane_idx < (eval_config_main.NUM_LANES - 1)
        final_action_to_execute = 0

        if monitoring_change:
            monitor_steps_left -= 1
            current_vehicle_info = env.get_vehicle_info()
            current_pos, current_lane_after_step = current_vehicle_info['pos'], current_vehicle_info['lane']
            lateral_dist = abs(current_pos[1] - monitor_start_pos[1]) if current_pos and monitor_start_pos else 0.0
            change_succeeded_criteria = lateral_dist >= eval_config_main.FORCE_CHANGE_SUCCESS_DIST
            if env.collision_occurred:
                forced_executed_collision += 1; monitoring_change = False; done = True
            elif change_succeeded_criteria:
                forced_executed_safe += 1; monitoring_change = False
            elif monitor_steps_left <= 0:
                monitoring_change = False  # Timeout
            if monitoring_change: final_action_to_execute = 0

        if not monitoring_change:
            is_forced_attempt_step = (step_count > 0 and step_count % eval_config_main.FORCE_CHANGE_INTERVAL_STEPS == 0)
            if is_forced_attempt_step:
                possible_proposals = [p for p, c in [(1, can_go_left), (2, can_go_right)] if c]
                if possible_proposals:
                    proposed_forced_action = random.choice(possible_proposals)
                    forced_attempts += 1
                    intended_action = get_dqn_model_action(model, state_norm, current_lane_idx, train_config,
                                                           eval_config_main.NUM_LANES, device)
                    if intended_action == proposed_forced_action:
                        forced_agreed += 1;
                        final_action_to_execute = intended_action;
                        agreed_lane_changes += 1
                        monitoring_change = True;
                        monitor_steps_left = eval_config_main.FORCE_CHANGE_MONITOR_STEPS
                        monitor_target_action = final_action_to_execute;
                        monitor_start_lane = current_lane_idx
                        monitor_start_pos = env.get_vehicle_info()['pos']
                    else:
                        forced_disagreed += 1; final_action_to_execute = 0
                else:
                    final_action_to_execute = 0
            else:
                final_action_to_execute = 0

        all_actions_ep.append(final_action_to_execute)
        if done: break
        next_state_raw, done = env.step(final_action_to_execute)
        state_raw = next_state_raw;
        step_count += 1
        vehicle_info = env.get_vehicle_info()
        if vehicle_info['lane'] != -1:
            all_speeds_ep.append(vehicle_info['speed']);
            all_lanes_ep.append(vehicle_info['lane'])
            current_front_dist = vehicle_info['front_dist'] if vehicle_info['front_dist'] is not None and np.isfinite(
                vehicle_info['front_dist']) else env.base_train_config.max_distance
            all_front_dists_ep.append(current_front_dist);
            min_front_dist_episode = min(min_front_dist_episode, current_front_dist)
            last_valid_vehicle_info = vehicle_info
        else:
            all_speeds_ep.append(0);
            all_lanes_ep.append(-1);
            all_front_dists_ep.append(env.base_train_config.max_distance)
            if not done: done = True
        if env.collision_occurred and not done: done = True

    avg_speed = np.mean([s for s in all_speeds_ep if s is not None]) if all_speeds_ep else 0.0
    total_dist = last_valid_vehicle_info['dist'] if last_valid_vehicle_info and last_valid_vehicle_info[
        'lane'] != -1 else 0.0
    result = EpisodeResult(step_count, env.collision_occurred, avg_speed, total_dist, forced_attempts, forced_agreed,
                           forced_disagreed, forced_executed_safe, forced_executed_collision, agreed_lane_changes,
                           min_front_dist_episode)
    return result, all_speeds_ep, all_lanes_ep, all_actions_ep, all_front_dists_ep


#####################################
# 主评估脚本                       #
#####################################
def main():
    eval_config = EvalConfig()
    # Load training configurations
    try:
        train_config_a = DQNYH_Config()
    except Exception as e:
        print(f"Error loading DQNYH_Config: {e}"); sys.exit(1)
    try:
        train_config_b = DQNNEW_Config()
    except Exception as e:
        print(f"Error loading DQNNEW_Config: {e}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    print(f"Using device: {device}")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evaluation_compare_results_{timestamp}";
    os.makedirs(output_dir, exist_ok=True);
    print(f"Output will be saved in: {output_dir}")

    print("\n--- Checking Files ---")
    for p in [eval_config.MODEL_A_PATH, eval_config.MODEL_B_PATH, eval_config.EVAL_SUMO_CONFIG, "new.net.xml",
              "new.rou.xml"]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}"); sys.exit(1)
        else:
            print(f"Found: {p}")

    print("\n--- Loading Models ---")
    model_a = load_dqn_model(eval_config.MODEL_A_PATH, train_config_a, device)
    model_b = load_dqn_model(eval_config.MODEL_B_PATH, train_config_b, device)

    obs_normalizer_a = RunningMeanStd(
        shape=(train_config_a.state_dim,)) if train_config_a.normalize_observations else None
    obs_normalizer_b = RunningMeanStd(
        shape=(train_config_b.state_dim,)) if train_config_b.normalize_observations else None
    if obs_normalizer_a or obs_normalizer_b:
        print("Initializing (frozen) normalizers. NOTE: Ideally, load saved training normalizer stats.")
        # Add placeholder for loading actual normalizer stats if available, e.g.:
        # if obs_normalizer_a and os.path.exists("dqnyh_normalizer_stats.pth"):
        #     stats = torch.load("dqnyh_normalizer_stats.pth")
        #     obs_normalizer_a.mean = stats['mean']; obs_normalizer_a.var = stats['var']

    # Use train_config_a for base env parameters like state_dim, assuming they are consistent
    env = EvaluationEnv(eval_config, train_config_a, eval_config.EVAL_SEED)

    print(f"\n--- Running Evaluation ({eval_config.EVAL_EPISODES} episodes/model) ---")
    results_model_a_list, results_model_b_list = [], []
    model_a_all_metrics = {'speeds': [], 'lanes': [], 'actions': [], 'front_dists': []}
    model_b_all_metrics = {'speeds': [], 'lanes': [], 'actions': [], 'front_dists': []}

    models_to_eval = [
        (eval_config.MODEL_A_NAME, model_a, train_config_a, obs_normalizer_a, results_model_a_list,
         model_a_all_metrics),
        (eval_config.MODEL_B_NAME, model_b, train_config_b, obs_normalizer_b, results_model_b_list,
         model_b_all_metrics),
    ]

    for i in tqdm(range(eval_config.EVAL_EPISODES), desc="Evaluating Episodes Total"):
        episode_seed = eval_config.EVAL_SEED + i
        print(f"\n--- Overall Episode {i + 1}/{eval_config.EVAL_EPISODES} (Seed: {episode_seed}) ---")
        for model_name, model_obj, tr_config, obs_norm, results_list_ref, all_metrics_ref in models_to_eval:
            print(f"Evaluating {model_name}...")
            env.sumo_seed = episode_seed  # Ensure same traffic for fair comparison within an episode
            try:
                ep_result, speeds, lanes, actions, f_dists = evaluate_episode(model_obj, tr_config, env, obs_norm,
                                                                              device, eval_config)
                if ep_result.steps == 0 and ep_result.collided:
                    print(f"{model_name} episode failed immediately.")
                else:
                    results_list_ref.append(ep_result)
                    all_metrics_ref['speeds'].extend(speeds);
                    all_metrics_ref['lanes'].extend(lanes)
                    all_metrics_ref['actions'].extend(actions);
                    all_metrics_ref['front_dists'].extend(f_dists)
                print(
                    f"{model_name} Ep {i + 1} Result: Steps={ep_result.steps}, Collided={ep_result.collided}, AgreedLC={ep_result.agreed_lane_changes}, Disagreed={ep_result.forced_disagreed}")
            except (ConnectionError, RuntimeError, traci.exceptions.TraCIException, KeyboardInterrupt) as e:
                print(f"\nError during {model_name} evaluation episode {i + 1}: {e}. Closing env and skipping.")
                time.sleep(0.5);
                env._close();
                time.sleep(1);
                break  # Break from inner loop (models) for this episode
            except Exception as e_other:
                print(f"\nUnexpected error during {model_name} evaluation episode {i + 1}: {e_other}");
                traceback.print_exc()
                time.sleep(0.5);
                env._close();
                time.sleep(1);
                break

    env._close();
    print("\n--- Evaluation Finished ---")

    # --- Aggregate and Compare Results ---
    if not results_model_a_list and not results_model_b_list: print("No results collected. Exiting."); return

    results_summary = {eval_config.MODEL_A_NAME: {}, eval_config.MODEL_B_NAME: {}}
    metrics_definitions = [
        ('total_episodes', '总评估回合数', 'd', True, None, None),
        ('avg_steps', '平均步数/回合', '.1f', True, 'std_steps', 'median_steps'),
        ('std_steps', '步数标准差', '.1f', False, None, None), ('median_steps', '步数中位数', '.1f', True, None, None),
        ('avg_speed_episode', '每回合平均速度 (m/s)', '.2f', True, 'std_speed_episode', None),
        ('median_speed_all', '所有步速度中位数 (m/s)', '.2f', True, None, None),
        ('avg_dist', '平均距离/回合 (m)', '.1f', True, 'std_dist', 'median_dist'),
        ('std_dist', '距离标准差 (m)', '.1f', False, None, None),
        ('median_dist', '距离中位数 (m)', '.1f', True, None, None),
        ('collision_rate', '碰撞率 (%)', '.1f', False, None, None),
        ('median_min_front_dist', '最小前车距离中位数 (m)', '.1f', True, None, None),
        ('avg_agreed_lc', '平均同意换道次数/回合', '.1f', None, None, None),
        ('total_forced_attempts', '强制换道尝试总数', 'd', None, None, None),
        ('forced_agreement_rate', '强制换道: 同意率 (%)', '.1f', True, None, None),
        ('forced_disagreement_rate', '强制换道: 不同意率 (%)', '.1f', False, None, None),
        ('forced_execution_success_rate_agreed', '执行成功率 (% 同意的)', '.1f', True, None, None),
        ('forced_execution_collision_rate_agreed', '执行碰撞率 (% 同意的)', '.1f', False, None, None),
    ]

    for model_name_key, res_list, all_mets in [
        (eval_config.MODEL_A_NAME, results_model_a_list, model_a_all_metrics),
        (eval_config.MODEL_B_NAME, results_model_b_list, model_b_all_metrics)
    ]:
        current_summary = results_summary[model_name_key]
        total_eps = len(res_list)
        current_summary['total_episodes'] = total_eps
        if total_eps == 0: print(f"Warning: No results for {model_name_key}."); continue

        steps_l = [r.steps for r in res_list];
        speeds_avg_l = [r.avg_speed for r in res_list];
        dists_l = [r.total_dist for r in res_list]
        min_front_d_l = [r.min_front_dist for r in res_list];
        collided_l = [r.collided for r in res_list];
        agreed_lc_l = [r.agreed_lane_changes for r in res_list]

        current_summary['avg_steps'] = np.mean(steps_l);
        current_summary['std_steps'] = np.std(steps_l);
        current_summary['median_steps'] = np.median(steps_l)
        current_summary['collision_rate'] = np.mean(collided_l) * 100
        current_summary['avg_speed_episode'] = np.mean(speeds_avg_l);
        current_summary['std_speed_episode'] = np.std(speeds_avg_l)
        current_summary['median_speed_all'] = np.median(all_mets['speeds']) if all_mets['speeds'] else 0
        current_summary['avg_dist'] = np.mean(dists_l);
        current_summary['std_dist'] = np.std(dists_l);
        current_summary['median_dist'] = np.median(dists_l)
        current_summary['avg_agreed_lc'] = np.mean(agreed_lc_l)
        current_summary['median_min_front_dist'] = np.median(
            min_front_d_l) if min_front_d_l else env.base_train_config.max_distance

        tot_f_attempts = sum(r.forced_attempts for r in res_list);
        tot_f_agreed = sum(r.forced_agreed for r in res_list)
        tot_f_disagreed = sum(r.forced_disagreed for r in res_list);
        tot_f_exec_safe = sum(r.forced_executed_safe for r in res_list)
        tot_f_exec_coll = sum(r.forced_executed_collision for r in res_list)

        current_summary['total_forced_attempts'] = tot_f_attempts
        current_summary['forced_agreement_rate'] = (tot_f_agreed / tot_f_attempts * 100) if tot_f_attempts > 0 else 0
        current_summary['forced_disagreement_rate'] = (
                    tot_f_disagreed / tot_f_attempts * 100) if tot_f_attempts > 0 else 0
        current_summary['forced_execution_success_rate_agreed'] = (
                    tot_f_exec_safe / tot_f_agreed * 100) if tot_f_agreed > 0 else 0
        current_summary['forced_execution_collision_rate_agreed'] = (
                    tot_f_exec_coll / tot_f_agreed * 100) if tot_f_agreed > 0 else 0

        current_summary['step_level_aggregates'] = {}
        lane_counts = Counter(l for l in all_mets['lanes'] if l >= 0);
        total_valid_lane_steps = sum(lane_counts.values())
        current_summary['step_level_aggregates']['lane_occupancy_percent'] = {
            f"lane_{l}": (c / total_valid_lane_steps * 100) if total_valid_lane_steps > 0 else 0 for l, c in
            lane_counts.items()}
        action_counts = Counter(all_mets['actions']);
        total_actions = len(all_mets['actions'])
        current_summary['step_level_aggregates']['action_distribution_percent'] = {
            f"action_{a}": (c / total_actions * 100) if total_actions > 0 else 0 for a, c in action_counts.items()}

    print("\n--- Results Comparison (Text) ---")
    comparison_lines = []
    header1 = f"{'Metric':<45} | {eval_config.MODEL_A_NAME:<30} | {eval_config.MODEL_B_NAME:<30}"
    header2 = "-" * len(header1)
    comparison_lines.append(header1);
    comparison_lines.append(header2);
    print(header1);
    print(header2)
    processed_keys = set()
    for key, name, fmt, _, std_key, median_key in metrics_definitions:
        if key in processed_keys: continue
        val_a = results_summary.get(eval_config.MODEL_A_NAME, {}).get(key, 'N/A')
        val_b = results_summary.get(eval_config.MODEL_B_NAME, {}).get(key, 'N/A')
        str_a = format(val_a, fmt) if isinstance(val_a, (int, float)) else str(val_a)
        str_b = format(val_b, fmt) if isinstance(val_b, (int, float)) else str(val_b)
        current_name = name
        precision = int(fmt[-2]) if len(fmt) >= 2 and fmt[-1] == 'f' else 1

        if std_key and std_key not in processed_keys:
            std_a = results_summary.get(eval_config.MODEL_A_NAME, {}).get(std_key);
            std_b = results_summary.get(eval_config.MODEL_B_NAME, {}).get(std_key)
            if isinstance(std_a, (int, float)): str_a += f" (±{std_a:.{precision}f})"
            if isinstance(std_b, (int, float)): str_b += f" (±{std_b:.{precision}f})"
            processed_keys.add(std_key)

        if median_key and median_key not in processed_keys:  # Simplified median reporting
            med_a = results_summary.get(eval_config.MODEL_A_NAME, {}).get(median_key, 'N/A')
            med_b = results_summary.get(eval_config.MODEL_B_NAME, {}).get(median_key, 'N/A')
            med_str_a = format(med_a, f'.{precision}f') if isinstance(med_a, (int, float)) else 'N/A'
            med_str_b = format(med_b, f'.{precision}f') if isinstance(med_b, (int, float)) else 'N/A'
            # If std was already added, replace the closing parenthesis, otherwise append.
            if std_key in processed_keys:
                if str_a.endswith(")"):
                    str_a = str_a[:-1] + f", Med: {med_str_a})"
                else:
                    str_a += f" (Med: {med_str_a})"
                if str_b.endswith(")"):
                    str_b = str_b[:-1] + f", Med: {med_str_b})"
                else:
                    str_b += f" (Med: {med_str_b})"
            else:
                str_a += f" (Med: {med_str_a})"
                str_b += f" (Med: {med_str_b})"
            processed_keys.add(median_key)

        line = f"{current_name:<45} | {str_a:<30} | {str_b:<30}";
        comparison_lines.append(line);
        print(line)
        processed_keys.add(key)

    comparison_lines.append("-" * len(header1));
    print("-" * len(header1))
    print("Step-Level Aggregates:");
    comparison_lines.append("Step-Level Aggregates:")
    for lane_i in range(eval_config.NUM_LANES):
        name = f"  车道 {lane_i} 占用率 (%)"
        perc_a = results_summary.get(eval_config.MODEL_A_NAME, {}).get('step_level_aggregates', {}).get(
            'lane_occupancy_percent', {}).get(f"lane_{lane_i}", 0.0)
        perc_b = results_summary.get(eval_config.MODEL_B_NAME, {}).get('step_level_aggregates', {}).get(
            'lane_occupancy_percent', {}).get(f"lane_{lane_i}", 0.0)
        line = f"{name:<45} | {perc_a:.1f}%{'':<25} | {perc_b:.1f}%{'':<25}";
        comparison_lines.append(line);
        print(line)
    comparison_lines.append("-" * len(header1));
    print("-" * len(header1))
    action_names = {0: "保持车道", 1: "向左变道", 2: "向右变道"}
    for action_i in range(3):
        name = f"  执行动作 '{action_names[action_i]}' (%)"
        perc_a = results_summary.get(eval_config.MODEL_A_NAME, {}).get('step_level_aggregates', {}).get(
            'action_distribution_percent', {}).get(f"action_{action_i}", 0.0)
        perc_b = results_summary.get(eval_config.MODEL_B_NAME, {}).get('step_level_aggregates', {}).get(
            'action_distribution_percent', {}).get(f"action_{action_i}", 0.0)
        line = f"{name:<45} | {perc_a:.1f}%{'':<25} | {perc_b:.1f}%{'':<25}";
        comparison_lines.append(line);
        print(line)
    comparison_lines.append(header2);
    print(header2)

    text_results_filename = os.path.join(output_dir, f"evaluation_compare_summary_{timestamp}.txt")
    with open(text_results_filename, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Run: {timestamp}\nMode: Forced Changes Only (Agent Agrees)\n")
        f.write(f"Model A ({eval_config.MODEL_A_NAME}): {os.path.basename(eval_config.MODEL_A_PATH)}\n")
        f.write(f"Model B ({eval_config.MODEL_B_NAME}): {os.path.basename(eval_config.MODEL_B_PATH)}\n")
        f.write(f"SUMO: {eval_config.EVAL_SUMO_CONFIG}, Episodes/model: {eval_config.EVAL_EPISODES}\n\n")
        f.write("\n".join(comparison_lines))
    print(f"Text results saved: {text_results_filename}")

    print("\n--- Generating Comparison Plots ---")
    model_labels = [eval_config.MODEL_A_NAME, eval_config.MODEL_B_NAME]
    colors = ['deepskyblue', 'lightcoral']
    plot_metrics_bar = [
        ('avg_steps', '每回合平均步数', '步数', 'avg_steps', 'std_steps'),
        ('avg_speed_episode', '每回合平均速度', '速度 (m/s)', 'avg_speed_per_episode', 'std_speed_episode'),
        ('collision_rate', '总碰撞率', '比率 (%)', 'collision_rate', None),
        ('avg_dist', '平均行驶距离', '距离 (m)', 'avg_dist', 'std_dist'),
        ('avg_agreed_lc', '平均同意换道次数', '次数', 'avg_agreed_lc', None),
        ('forced_agreement_rate', '强制换道: 同意率', '比率 (%)', 'forced_agreement_rate', None),
        ('forced_disagreement_rate', '强制换道: 不同意率', '比率 (%)', 'forced_disagreement_rate', None),
        ('forced_execution_success_rate_agreed', '同意换道: 执行成功率', '比率 (%)', 'forced_exec_success_rate', None),
        ('forced_execution_collision_rate_agreed', '同意换道: 执行碰撞率', '比率 (%)', 'forced_exec_collision_rate',
         None),
    ]
    for key, title, ylabel, fname_suffix, std_key in plot_metrics_bar:
        plt.figure(figsize=(7, 6));
        plt.title(f'{title}\n({model_labels[0]} vs {model_labels[1]})', fontsize=14)
        val_a = results_summary.get(model_labels[0], {}).get(key);
        val_b = results_summary.get(model_labels[1], {}).get(key)
        if val_a is None or val_b is None: print(f"Skipping plot '{title}' due to missing data."); plt.close(); continue
        values = [val_a, val_b];
        std_devs = [0, 0];
        has_std = False
        if std_key:
            std_a = results_summary.get(model_labels[0], {}).get(std_key);
            std_b = results_summary.get(model_labels[1], {}).get(std_key)
            if isinstance(std_a, (int, float)) and np.isfinite(std_a) and isinstance(std_b,
                                                                                     (int, float)) and np.isfinite(
                    std_b):
                std_devs = [std_a, std_b];
                has_std = True
        fmt_str = next((m[2] for m in metrics_definitions if m[0] == key), ".1f");
        precision = int(fmt_str[-2]) if len(fmt_str) >= 2 and fmt_str[-1] == 'f' else 1
        bars = plt.bar(model_labels, values, color=colors, yerr=std_devs if has_std else None,
                       capsize=5 if has_std else 0)
        for i, bar in enumerate(bars):
            yval = bar.get_height();
            y_offset = yval + (std_devs[i] * 0.6 if has_std else 0) + max(values) * 0.02
            label_text = f"{yval:.{precision}f}" + ("%" if '%' in ylabel else "")
            plt.text(bar.get_x() + bar.get_width() / 2.0, y_offset, label_text, ha='center', va='bottom', fontsize=10)
        plt.ylabel(ylabel, fontsize=12);
        plt.xticks(fontsize=12);
        plt.yticks(fontsize=10)
        max_val_display = max(yval + (std_devs[i] if has_std else 0) for i, yval in enumerate(values))
        min_val_display = min(yval - (std_devs[i] if has_std else 0) for i, yval in enumerate(values))
        if '%' in ylabel:
            plt.ylim(0, max(1, max_val_display * 1.15, 105))
        elif max_val_display > 0:
            plt.ylim(bottom=max(0, min_val_display * 0.9), top=max_val_display * 1.15)
        elif max_val_display == 0:
            plt.ylim(-0.1, 1)
        else:
            plt.ylim(bottom=min_val_display * 1.15, top=max(0.1, max_val_display * 0.8))
        plt.grid(axis='y', linestyle='--', alpha=0.7);
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"plot_bar_{fname_suffix}_{timestamp}.png"));
        plt.close()
    print("Bar plots saved.")

    # Box plots
    for data_key, plot_title, y_label, fname_suffix_box in [
        ('speeds', '所有步骤的速度分布', '速度 (m/s)', 'speed_dist'),
        ('front_dists', '所有步骤的前车距离分布', '前车距离 (m)', 'front_dist_dist')
    ]:
        data_a = model_a_all_metrics[data_key];
        data_b = model_b_all_metrics[data_key]
        if data_a or data_b:
            plt.figure(figsize=(7, 5));
            plt.title(f'{plot_title}\n({model_labels[0]} vs {model_labels[1]})', fontsize=14)
            plot_data, current_labels = [], []
            if data_a: plot_data.append(data_a); current_labels.append(model_labels[0])
            if data_b: plot_data.append(data_b); current_labels.append(model_labels[1])
            if plot_data:
                plt.boxplot(plot_data, labels=current_labels, showfliers=True)
                plt.ylabel(y_label);
                plt.grid(axis='y', linestyle='--');
                plt.tight_layout()
                if 'dist' in fname_suffix_box: plt.ylim(bottom=0)  # Ensure distance starts at 0
                plt.savefig(os.path.join(output_dir, f"plot_box_{fname_suffix_box}_{timestamp}.png"));
                plt.close()
    print("Distribution plots saved.")

    # Lane Occupancy and Action Distribution plots (similar to pinggunew.py)
    for plot_type in ['lane_occupancy_percent', 'action_distribution_percent']:
        data_a = results_summary.get(model_labels[0], {}).get('step_level_aggregates', {}).get(plot_type)
        data_b = results_summary.get(model_labels[1], {}).get('step_level_aggregates', {}).get(plot_type)
        if data_a or data_b:
            if plot_type == 'lane_occupancy_percent':
                title = '车道占用率';
                labels = [f'车道 {i}' for i in range(eval_config.NUM_LANES)]
                keys = [f'lane_{i}' for i in range(eval_config.NUM_LANES)]
            else:  # action_distribution_percent
                title = '执行动作分布';
                labels = ['保持车道', '向左变道', '向右变道']
                keys = [f'action_{i}' for i in range(3)]
            plt.figure(figsize=(8, 5));
            plt.title(f'{title}\n({model_labels[0]} vs {model_labels[1]})', fontsize=14)
            perc_a = [data_a.get(k, 0.0) if data_a else 0.0 for k in keys]
            perc_b = [data_b.get(k, 0.0) if data_b else 0.0 for k in keys]
            x = np.arange(len(labels));
            width = 0.35
            rects1 = plt.bar(x - width / 2, perc_a, width, label=model_labels[0], color=colors[0])
            rects2 = plt.bar(x + width / 2, perc_b, width, label=model_labels[1], color=colors[1])
            plt.ylabel('百分比 (%)');
            plt.xticks(x, labels);
            plt.ylim(0, 105);
            plt.legend()
            plt.bar_label(rects1, padding=3, fmt='%.1f%%');
            plt.bar_label(rects2, padding=3, fmt='%.1f%%')
            plt.grid(axis='y', linestyle='--', alpha=0.7);
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"plot_bar_{plot_type.replace('_percent', '')}_{timestamp}.png"));
            plt.close()
    print("Step aggregate plots saved.")

    # Save JSON Data
    data_filename = os.path.join(output_dir, f"evaluation_compare_data_{timestamp}.json")
    results_summary['eval_config_details'] = {k: v for k, v in vars(EvalConfig).items() if not k.startswith('__')}
    # Add raw episode results to the summary for detailed analysis if needed
    results_summary[model_labels[0]]['raw_episode_results'] = [r._asdict() for r in results_model_a_list]
    results_summary[model_labels[1]]['raw_episode_results'] = [r._asdict() for r in results_model_b_list]

    class NpEncoder(json.JSONEncoder):  # Handles numpy types for JSON dump
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else str(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.bool_): return bool(obj)
            return super(NpEncoder, self).default(obj)

    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        print(f"Detailed data saved: {data_filename}")
    except Exception as e:
        print(f"Error saving JSON data: {e}")


if __name__ == "__main__":
    main()
