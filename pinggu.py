#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版SUMO自动驾驶模型评估脚本
用于评估DQN和PPO两种算法训练的车道变更模型
"""

import os
import sys
import time
import subprocess
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import argparse
import signal
import torch
import tensorflow as tf
from collections import defaultdict
import warnings

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 检查SUMO_HOME环境变量
if 'SUMO_HOME' not in os.environ:
    sys.exit("请设置环境变量'SUMO_HOME'")
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import traci
import sumolib

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# 配置类
class Config:
    sumo_binary = sumolib.checkBinary("sumo")  # 可改为"sumo-gui"进行可视化
    config_path = "4.sumocfg"  # 与DQN训练一致，需确保路径正确

    num_episodes = 30
    max_steps = 1000
    ego_vehicle_id = "eval_vehicle"
    port_range = (25000, 26000)
    step_length = 0.1

    dqn_model_path = None
    ppo_model_path = None

    dqn_state_dim = 22  # 与DQN训练一致
    ppo_state_dim = 10  # 与PPO训练一致
    action_dim = 3  # 0:保持 1:左变 2:右变

    metrics = [
        'avg_speed', 'max_speed', 'avg_acceleration', 'max_acceleration',
        'avg_jerk', 'lane_changes', 'collisions', 'travel_time',
        'travel_distance', 'safety_violations', 'center_lane_time',
        'lane_distribution', 'efficiency_score', 'safety_score',
        'comfort_score', 'overall_score'
    ]


# 工具函数
def kill_process_by_port(port):
    if os.name == 'nt':
        try:
            subprocess.run(f"FOR /F \"tokens=5\" %P IN ('netstat -ano ^| findstr {port}') DO taskkill /F /PID %P",
                           shell=True)
        except:
            pass
    else:
        try:
            subprocess.run(f"kill -9 $(lsof -ti tcp:{port})", shell=True)
        except:
            pass


def is_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = True
    try:
        sock.bind(('127.0.0.1', port))
    except:
        result = False
    sock.close()
    return result


def find_available_port(start_port=20000, end_port=30000):
    for port in range(start_port, end_port):
        if is_port_available(port):
            return port
    raise RuntimeError("无法找到可用端口")


def kill_all_sumo_instances():
    if os.name == 'nt':
        os.system("taskkill /f /im sumo.exe >nul 2>&1")
        os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")
    else:
        os.system("pkill -f sumo")
        os.system("pkill -f sumo-gui")


# PPO模型定义（与训练代码一致）
class PPOModel(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_size=512, action_dim=3):
        super(PPOModel, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim),
            torch.nn.Softmax(dim=-1)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


# SUMO评估环境
class SUMOEvalEnv:
    def __init__(self, ego_id=None, episode_id=0):
        self.ego_id = ego_id or f"{Config.ego_vehicle_id}_{episode_id}"
        self.episode_id = episode_id
        self.connection_id = None
        self.step_length = Config.step_length
        self.max_steps = Config.max_steps
        self.reset_metrics()

    def reset_metrics(self):
        self.speed_history = []
        self.accel_history = [0]
        self.jerk_history = [0, 0]
        self.lane_history = []
        self.lane_changes = 0
        self.collisions = 0
        self.safety_violations = 0
        self.start_time = 0
        self.travel_distance = 0
        self.start_position = None
        self.prev_position = None
        self.prev_speed = 0
        self.prev_lane = -1
        self.step_count = 0

    def start(self):
        self.close()
        self.connection_id = f"sim_{self.episode_id}"
        sumo_cmd = [
            Config.sumo_binary,
            "-c", Config.config_path,
            "--num-clients", "1",
            "--start", "true",
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--collision.action", "warn",
            "--time-to-teleport", "-1",
            "--random", "true",
            "--seed", str(RANDOM_SEED + self.episode_id)
        ]
        attempts = 5
        for _ in range(attempts):
            try:
                traci.start(sumo_cmd, label=self.connection_id)
                traci.switch(self.connection_id)
                self.add_ego_vehicle()
                self.reset_metrics()
                self.start_time = traci.simulation.getTime()
                return True
            except Exception as e:
                print(f"启动SUMO失败，尝试重试。错误: {e}")
                self.close()
                time.sleep(1)
        print(f"在{attempts}次尝试后无法启动SUMO，放弃此评估回合")
        return False

    def add_ego_vehicle(self):
        if "ego_route" not in traci.route.getIDList():
            traci.route.add("ego_route", ["E0"])
        traci.vehicle.add(
            self.ego_id, "ego_route",
            typeID="car",
            departPos="0",
            departLane="random",
            departSpeed="max"
        )
        traci.vehicle.setSpeedMode(self.ego_id, 31)
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        for _ in range(10):
            traci.simulationStep()
            if self.ego_id in traci.vehicle.getIDList():
                self.start_position = traci.vehicle.getPosition(self.ego_id)
                self.prev_position = self.start_position
                self.prev_lane = traci.vehicle.getLaneIndex(self.ego_id)
                break

    def get_dqn_state(self):
        state = np.zeros(Config.dqn_state_dim)
        if self.ego_id not in traci.vehicle.getIDList():
            return state.reshape(1, Config.dqn_state_dim)

        try:
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            ego_lane_id = traci.vehicle.getLaneID(self.ego_id)
            lane_max_speed = traci.lane.getMaxSpeed(ego_lane_id)
            norm_speed = ego_speed / max(1.0, lane_max_speed)
            lane_one_hot = [0, 0, 0]
            if 0 <= ego_lane < 3:
                lane_one_hot[ego_lane] = 1

            leading_vehicle = [0, 100, 0]
            leader = traci.vehicle.getLeader(self.ego_id)
            if leader:
                lead_id, lead_dist = leader
                if lead_id:
                    leading_vehicle = [1, lead_dist, traci.vehicle.getSpeed(lead_id) - ego_speed]

            surrounding_vehicles = traci.vehicle.getIDList()
            following_vehicle = [0, 100, 0]
            left_leading = [0, 100, 0]
            left_following = [0, 100, 0]
            right_leading = [0, 100, 0]
            right_following = [0, 100, 0]

            for v_id in surrounding_vehicles:
                if v_id != self.ego_id:
                    v_lane = traci.vehicle.getLaneIndex(v_id)
                    v_pos = traci.vehicle.getLanePosition(v_id)
                    v_speed = traci.vehicle.getSpeed(v_id)
                    rel_pos = v_pos - ego_pos
                    if v_lane == ego_lane and rel_pos < 0:
                        dist = abs(rel_pos)
                        if dist < following_vehicle[1]:
                            following_vehicle = [1, dist, v_speed - ego_speed]
                    if ego_lane > 0 and v_lane == ego_lane - 1:
                        if rel_pos > 0 and rel_pos < left_leading[1]:
                            left_leading = [1, rel_pos, v_speed - ego_speed]
                        elif rel_pos < 0 and abs(rel_pos) < left_following[1]:
                            left_following = [1, abs(rel_pos), v_speed - ego_speed]
                    if ego_lane < 2 and v_lane == ego_lane + 1:
                        if rel_pos > 0 and rel_pos < right_leading[1]:
                            right_leading = [1, rel_pos, v_speed - ego_speed]
                        elif rel_pos < 0 and abs(rel_pos) < right_following[1]:
                            right_following = [1, abs(rel_pos), v_speed - ego_speed]

            state = [
                norm_speed,
                *lane_one_hot,
                *leading_vehicle,
                *following_vehicle,
                *left_leading,
                *left_following,
                *right_leading,
                *right_following
            ]
        except:
            pass

        return np.array(state).reshape(1, Config.dqn_state_dim)

    def get_ppo_state(self):
        state = np.zeros(Config.ppo_state_dim, dtype=np.float32)
        if self.ego_id not in traci.vehicle.getIDList():
            return state

        try:
            speed = traci.vehicle.getSpeed(self.ego_id)
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            state[0] = speed / 33.33
            state[1] = lane / 2.0

            ego_pos = traci.vehicle.getPosition(self.ego_id)
            ranges = {
                'front': (100.0, -1),
                'back': (100.0, -1),
                'left_front': (100.0, -1),
                'left_back': (100.0, -1),
                'right_front': (100.0, -1),
                'right_back': (100.0, -1)
            }

            for veh_id in traci.vehicle.getIDList():
                if veh_id == self.ego_id:
                    continue
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                dx = traci.vehicle.getPosition(veh_id)[0] - ego_pos[0]
                dy = traci.vehicle.getPosition(veh_id)[1] - ego_pos[1]
                distance = np.hypot(dx, dy)
                if veh_lane == lane - 1:
                    key = 'left_front' if dx > 0 else 'left_back'
                elif veh_lane == lane + 1:
                    key = 'right_front' if dx > 0 else 'right_back'
                elif veh_lane == lane:
                    key = 'front' if dx > 0 else 'back'
                else:
                    continue
                if distance < ranges[key][0]:
                    ranges[key] = (distance, veh_id)

            state[2] = ranges['front'][0] / 100.0
            state[3] = ranges['back'][0] / 100.0
            state[4] = ranges['left_front'][0] / 100.0
            state[5] = ranges['left_back'][0] / 100.0
            state[6] = ranges['right_front'][0] / 100.0
            state[7] = ranges['right_back'][0] / 100.0

            state[8] = state[1]
            state[9] = 1.0 if lane == 1 else 0.0
        except:
            pass

        return state

    def step_dqn(self, action):
        if self.ego_id not in traci.vehicle.getIDList():
            return -100, True

        old_lane = traci.vehicle.getLaneIndex(self.ego_id)
        collision = False

        try:
            if action == 1 and old_lane > 0:
                traci.vehicle.changeLane(self.ego_id, old_lane - 1, 5)  # 与DQN训练一致
            elif action == 2 and old_lane < 2:
                traci.vehicle.changeLane(self.ego_id, old_lane + 1, 5)

            current_speed = traci.vehicle.getSpeed(self.ego_id)
            lane_id = traci.vehicle.getLaneID(self.ego_id)
            max_lane_speed = traci.lane.getMaxSpeed(lane_id)
            desired_speed = min(max_lane_speed, current_speed + 1)
            traci.vehicle.setSpeed(self.ego_id, desired_speed)

            for _ in range(5):
                traci.simulationStep()
                self.update_metrics()
                if self.check_collision():
                    collision = True
                    break
                if self.ego_id not in traci.vehicle.getIDList():
                    break
        except:
            collision = True

        new_lane = traci.vehicle.getLaneIndex(self.ego_id) if self.ego_id in traci.vehicle.getIDList() else old_lane
        if new_lane != old_lane and new_lane != -1 and old_lane != -1:
            self.lane_changes += 1

        reward = -150 if collision else (1 if old_lane != new_lane else 0.1)
        done = collision or self.step_count >= self.max_steps
        if done and not collision and self.ego_id in traci.vehicle.getIDList():
            reward += 10

        return reward, done

    def step_ppo(self, action):
        if self.ego_id not in traci.vehicle.getIDList():
            return -50, True

        reward = 0.0
        done = False
        old_lane = traci.vehicle.getLaneIndex(self.ego_id)

        try:
            if action == 1 and old_lane > 0:
                traci.vehicle.changeLane(self.ego_id, old_lane - 1, 2)  # 与PPO训练一致
            elif action == 2 and old_lane < 2:
                traci.vehicle.changeLane(self.ego_id, old_lane + 1, 2)

            traci.simulationStep()
            self.update_metrics()

            if self.check_collision():
                return -50.0, True

            new_lane = traci.vehicle.getLaneIndex(self.ego_id)
            if new_lane != old_lane and new_lane != -1 and old_lane != -1:
                self.lane_changes += 1

            speed = traci.vehicle.getSpeed(self.ego_id)
            reward += (speed / 33.33) * 0.3
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            reward += (2 - abs(lane - 1)) * 0.2
            if action != 0:
                reward += 0.2
        except:
            done = True

        done = done or self.step_count >= self.max_steps
        return reward, done

    def update_metrics(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return

        try:
            speed = traci.vehicle.getSpeed(self.ego_id)
            position = traci.vehicle.getPosition(self.ego_id)
            lane = traci.vehicle.getLaneIndex(self.ego_id)

            accel = (speed - self.prev_speed) / max(0.1, self.step_length)
            jerk = (accel - self.accel_history[-1]) / max(0.1, self.step_length) if self.accel_history else 0

            self.speed_history.append(speed)
            self.accel_history.append(accel)
            self.jerk_history.append(jerk)
            self.lane_history.append(lane)

            if self.prev_position:
                dx = position[0] - self.prev_position[0]
                dy = position[1] - self.prev_position[1]
                self.travel_distance += np.hypot(dx, dy)

            if traci.vehicle.getLeader(self.ego_id):
                _, lead_dist = traci.vehicle.getLeader(self.ego_id)
                if lead_dist < speed * 1.5:
                    self.safety_violations += 1

            self.prev_speed = speed
            self.prev_position = position
            self.step_count += 1
        except:
            pass

    def check_collision(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return True
        collision_list = traci.simulation.getCollidingVehiclesIDList()
        if collision_list and self.ego_id in collision_list:
            self.collisions += 1
            return True
        return False

    def get_metrics(self):
        metrics = {
            'avg_speed': np.mean(self.speed_history) if self.speed_history else 0,
            'max_speed': np.max(self.speed_history) if self.speed_history else 0,
            'avg_acceleration': np.mean(np.abs(self.accel_history[1:])) if len(self.accel_history) > 1 else 0,
            'max_acceleration': np.max(np.abs(self.accel_history[1:])) if len(self.accel_history) > 1 else 0,
            'avg_jerk': np.mean(np.abs(self.jerk_history[2:])) if len(self.jerk_history) > 2 else 0,
            'lane_changes': self.lane_changes,
            'collisions': self.collisions,
            'travel_time': traci.simulation.getTime() - self.start_time if self.start_time > 0 else 0,
            'travel_distance': self.travel_distance,
            'safety_violations': self.safety_violations,
            'center_lane_time': self.lane_history.count(1) / len(self.lane_history) if self.lane_history else 0,
            'lane_distribution': {lane: self.lane_history.count(lane) / len(self.lane_history) for lane in
                                  set(self.lane_history)} if self.lane_history else {},
            'efficiency_score': 0,
            'safety_score': 0,
            'comfort_score': 0,
            'overall_score': 0
        }

        if metrics['avg_speed'] > 0:
            metrics['efficiency_score'] = min(10, (metrics['avg_speed'] / 33.33) * 10)
            if metrics['travel_time'] > 0:
                metrics['efficiency_score'] -= min(5, metrics['travel_time'] / 200)
            metrics['efficiency_score'] -= min(3, metrics['lane_changes'] * 0.2)

        metrics['safety_score'] = 10 - metrics['collisions'] * 10 - min(5, metrics['safety_violations'] * 0.1)
        metrics['comfort_score'] = 10 - min(5, metrics['avg_acceleration']) - min(5, metrics['avg_jerk'] * 2)
        metrics['overall_score'] = (
                0.4 * max(0, metrics['safety_score']) +
                0.4 * max(0, metrics['efficiency_score']) +
                0.2 * max(0, metrics['comfort_score'])
        )

        for score in ['safety_score', 'efficiency_score', 'comfort_score', 'overall_score']:
            metrics[score] = max(0, min(10, metrics[score]))

        return metrics

    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass
        time.sleep(0.2)


# DQN代理类
class DQNAgent:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def act(self, state):
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])


# PPO代理类
class PPOAgent:
    def __init__(self, model_path):
        self.model = PPOModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            probs, _ = self.model(state_tensor)
            lane = int(state[1] * 2)
            mask = torch.ones(3)
            if lane == 0:
                mask[1] = 0.0
            elif lane == 2:
                mask[2] = 0.0
            probs = probs * mask
            if probs.sum() > 0:
                probs = probs / probs.sum()
                return torch.argmax(probs).item()
            return 0


# 评估函数
def evaluate_agent(agent_type, agent, num_episodes):
    print(f"开始评估 {agent_type} 模型...")
    all_metrics = []

    for episode in tqdm(range(num_episodes), desc=f"评估 {agent_type}"):
        env = SUMOEvalEnv(episode_id=episode)
        if not env.start():
            continue

        done = False
        total_reward = 0

        while not done:
            if agent_type == 'DQN':
                state = env.get_dqn_state()
                action = agent.act(state)
                reward, done = env.step_dqn(action)
            else:
                state = env.get_ppo_state()
                action = agent.act(state)
                reward, done = env.step_ppo(action)
            total_reward += reward

            if env.step_count >= Config.max_steps:
                done = True

        metrics = env.get_metrics()
        metrics['episode'] = episode
        metrics['total_reward'] = total_reward
        all_metrics.append(metrics)
        env.close()

        if (episode + 1) % 10 == 0 or episode == num_episodes - 1:
            avg_reward = np.mean([m['total_reward'] for m in all_metrics[-10:]])
            avg_score = np.mean([m['overall_score'] for m in all_metrics[-10:]])
            print(f"回合 {episode + 1}/{num_episodes}, 平均奖励: {avg_reward:.2f}, 平均得分: {avg_score:.2f}")

    avg_metrics = {}
    for metric in Config.metrics + ['total_reward']:
        if metric == 'lane_distribution':
            lane_dist = {}
            for m in all_metrics:
                for lane, pct in m.get('lane_distribution', {}).items():
                    lane_dist[lane] = lane_dist.get(lane, 0) + pct
            total = sum(lane_dist.values())
            avg_metrics['lane_distribution'] = {lane: val / total for lane, val in
                                                lane_dist.items()} if total > 0 else {}
        else:
            values = [m.get(metric, 0) for m in all_metrics]
            avg_metrics[metric] = np.mean(values) if values else 0

    return avg_metrics, all_metrics


# 创建比较图表
def create_comparison_charts(dqn_metrics, ppo_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    dqn_avg, dqn_all = dqn_metrics
    ppo_avg, ppo_all = ppo_metrics

    metrics_df = pd.DataFrame({
        'DQN': [dqn_avg[m] for m in Config.metrics if m != 'lane_distribution'],
        'PPO': [ppo_avg[m] for m in Config.metrics if m != 'lane_distribution'],
        'Metric': [m for m in Config.metrics if m != 'lane_distribution']
    })

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    speed_metrics = ['avg_speed', 'max_speed', 'travel_distance']
    speed_df = metrics_df[metrics_df['Metric'].isin(speed_metrics)]
    speed_melted = pd.melt(speed_df, id_vars=['Metric'], value_vars=['DQN', 'PPO'])
    sns.barplot(x='Metric', y='value', hue='variable', data=speed_melted)
    plt.title('速度和距离指标比较')
    plt.ylabel('值')

    plt.subplot(3, 1, 2)
    safety_metrics = ['collisions', 'safety_violations', 'lane_changes']
    safety_df = metrics_df[metrics_df['Metric'].isin(safety_metrics)]
    safety_melted = pd.melt(safety_df, id_vars=['Metric'], value_vars=['DQN', 'PPO'])
    sns.barplot(x='Metric', y='value', hue='variable', data=safety_melted)
    plt.title('安全指标比较')
    plt.ylabel('次数')

    plt.subplot(3, 1, 3)
    score_metrics = ['safety_score', 'efficiency_score', 'comfort_score', 'overall_score']
    score_df = metrics_df[metrics_df['Metric'].isin(score_metrics)]
    score_melted = pd.melt(score_df, id_vars=['Metric'], value_vars=['DQN', 'PPO'])
    sns.barplot(x='Metric', y='value', hue='variable', data=score_melted)
    plt.title('综合得分比较')
    plt.ylabel('得分 (0-10)')
    plt.ylim(0, 10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    dqn_lane_dist = dqn_avg.get('lane_distribution', {})
    ppo_lane_dist = ppo_avg.get('lane_distribution', {})
    dqn_lanes = sorted(dqn_lane_dist.keys())
    ppo_lanes = sorted(ppo_lane_dist.keys())
    dqn_values = [dqn_lane_dist.get(lane, 0) for lane in dqn_lanes]
    ppo_values = [ppo_lane_dist.get(lane, 0) for lane in ppo_lanes]

    width = 0.35
    plt.bar(np.array(dqn_lanes) - width / 2, dqn_values, width, label='DQN')
    plt.bar(np.array(ppo_lanes) + width / 2, ppo_values, width, label='PPO')
    plt.xlabel('车道')
    plt.ylabel('使用比例')
    plt.title('车道使用分布比较')
    plt.xticks(range(3))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lane_distribution.png")
    plt.close()

    categories = ['速度', '安全性', '效率', '舒适度', '车道保持']
    dqn_values = [
        min(10, (dqn_avg['avg_speed'] / 33.33) * 10),
        dqn_avg['safety_score'],
        dqn_avg['efficiency_score'],
        dqn_avg['comfort_score'],
        min(10, dqn_avg['center_lane_time'] * 10)
    ]
    ppo_values = [
        min(10, (ppo_avg['avg_speed'] / 33.33) * 10),
        ppo_avg['safety_score'],
        ppo_avg['efficiency_score'],
        ppo_avg['comfort_score'],
        min(10, ppo_avg['center_lane_time'] * 10)
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    dqn_values += dqn_values[:1]
    ppo_values += ppo_values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, dqn_values, 'b-', linewidth=2, label='DQN')
    ax.fill(angles, dqn_values, 'b', alpha=0.1)
    ax.plot(angles, ppo_values, 'r-', linewidth=2, label='PPO')
    ax.fill(angles, ppo_values, 'r', alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 10)
    ax.set_title("模型性能雷达图")
    ax.legend(loc='upper right')
    plt.savefig(f"{output_dir}/radar_comparison.png")
    plt.close()

    comparison_df = pd.DataFrame({
        'Metric': [m for m in Config.metrics if m != 'lane_distribution'],
        'DQN': [dqn_avg[m] for m in Config.metrics if m != 'lane_distribution'],
        'PPO': [ppo_avg[m] for m in Config.metrics if m != 'lane_distribution'],
        'Difference': [dqn_avg[m] - ppo_avg[m] for m in Config.metrics if m != 'lane_distribution'],
        'Percent_Difference': [
            (dqn_avg[m] - ppo_avg[m]) / max(0.001, ppo_avg[m]) * 100
            for m in Config.metrics if m != 'lane_distribution'
        ]
    })

    comparison_df.to_csv(f"{output_dir}/comparison_results.csv", index=False)
    pd.DataFrame(dqn_all).to_csv(f"{output_dir}/dqn_detailed_results.csv", index=False)
    pd.DataFrame(ppo_all).to_csv(f"{output_dir}/ppo_detailed_results.csv", index=False)

    return comparison_df


# 主函数
def main():
    parser = argparse.ArgumentParser(description='评估DQN和PPO车道变更模型')
    parser.add_argument('--dqn-model', type=str, required=True, help='DQN模型路径 (.keras)')
    parser.add_argument('--ppo-model', type=str, required=True, help='PPO模型路径 (.pth)')
    parser.add_argument('--episodes', type=int, default=Config.num_episodes, help='评估回合数')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='输出目录')
    parser.add_argument('--visual', action='store_true', help='使用SUMO-GUI可视化')

    args = parser.parse_args()

    Config.dqn_model_path = args.dqn_model
    Config.ppo_model_path = args.ppo_model
    Config.num_episodes = args.episodes
    if args.visual:
        Config.sumo_binary = sumolib.checkBinary('sumo-gui')

    os.makedirs(args.output_dir, exist_ok=True)
    kill_all_sumo_instances()
    time.sleep(1)

    def signal_handler(sig, frame):
        print('接收到中断信号，正在清理...')
        kill_all_sumo_instances()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"开始评估，每种模型 {Config.num_episodes} 个回合...")
    try:
        print("加载DQN模型...")
        dqn_agent = DQNAgent(Config.dqn_model_path)

        print("加载PPO模型...")
        ppo_agent = PPOAgent(Config.ppo_model_path)

        print("评估DQN模型...")
        dqn_metrics = evaluate_agent('DQN', dqn_agent, Config.num_episodes)

        kill_all_sumo_instances()
        time.sleep(1)

        print("评估PPO模型...")
        ppo_metrics = evaluate_agent('PPO', ppo_agent, Config.num_episodes)

        print("创建比较图表...")
        comparison_df = create_comparison_charts(dqn_metrics, ppo_metrics, args.output_dir)

        print("\n评估完成！结果摘要:")
        print("-" * 60)
        print(f"{'指标':<20} {'DQN':<10} {'PPO':<10} {'差值':<10}")
        print("-" * 60)
        key_metrics = ['avg_speed', 'collisions', 'lane_changes', 'safety_score',
                       'efficiency_score', 'comfort_score', 'overall_score']
        for metric in key_metrics:
            dqn_value = dqn_metrics[0][metric]
            ppo_value = ppo_metrics[0][metric]
            diff = dqn_value - ppo_value
            print(f"{metric:<20} {dqn_value:<10.2f} {ppo_value:<10.2f} {diff:<10.2f}")
        print("-" * 60)
        print(f"综合得分: DQN: {dqn_metrics[0]['overall_score']:.2f}, PPO: {ppo_metrics[0]['overall_score']:.2f}")
        winner = "DQN" if dqn_metrics[0]['overall_score'] > ppo_metrics[0]['overall_score'] else "PPO"
        print(f"在此评估中，{winner} 模型整体表现更好。")
        print(f"详细结果保存在 {args.output_dir}/ 目录中")
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        kill_all_sumo_instances()


if __name__ == "__main__":
    main()