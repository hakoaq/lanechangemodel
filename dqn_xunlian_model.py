# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime
import traci
import sumolib

# 设置GPU内存动态增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 检查SUMO_HOME环境变量
if 'SUMO_HOME' not in os.environ:
    sys.exit("请设置环境变量'SUMO_HOME'")
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# SUMO配置
SUMO_CONFIG_PATH = "a.sumocfg"
SUMO_BINARY = sumolib.checkBinary("sumo")  # 使用"sumo-gui"可进行可视化

# DQN智能体
class LaneChangeDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # 增加记忆库容量
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # 减小衰减率，延长探索
        self.learning_rate = 0.0005  # 调整学习率
        self.batch_size = 128  # 增大批量大小
        self.train_start = 1000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))  # 增加神经元数量
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))  # 增加一层
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))  # 增加深度
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # 添加学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9
        )
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr_schedule))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.train_start:
            return 0
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states = np.zeros((len(minibatch), self.state_size))
        next_states = np.zeros((len(minibatch), self.state_size))
        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = state[0]
            next_states[i] = next_state[0]
        targets = self.model.predict(states, verbose=0)
        next_state_values = self.target_model.predict(next_states, verbose=0)
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            targets[i][action] = reward if done else reward + self.gamma * np.max(next_state_values[i])
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss'][0]

    def save(self, name):
        if not name.endswith('.keras'):
            name += '.keras'
        self.model.save(name)

# SUMO环境
class SUMOEnvironment:
    def __init__(self, ego_vehicle_id, sumo_config_path, sumo_binary):
        self.ego_vehicle_id = ego_vehicle_id
        self.sumo_config_path = sumo_config_path
        self.sumo_binary = sumo_binary
        self.max_steps = 2000  # 增加最大步数
        self.current_step = 0
        self.sim_step_length = 0.1
        self.state_size = 22
        self.action_size = 3
        self.safe_distance_front = 15
        self.safe_distance_rear = 10

    def start_simulation(self):
        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config_path, "--start", "--step-length", str(self.sim_step_length)]
        if self.sumo_binary == "sumo":
            sumo_cmd.extend(["--no-step-log", "--no-warnings"])
        traci.start(sumo_cmd)
        traci.route.add("ego_route", ["E0"])
        traci.vehicle.add(self.ego_vehicle_id, "ego_route", typeID="car")
        traci.vehicle.setSpeedMode(self.ego_vehicle_id, 31)
        traci.vehicle.setLaneChangeMode(self.ego_vehicle_id, 0)
        traci.vehicle.moveTo(self.ego_vehicle_id, "E0_1", 5)
        traci.vehicle.setSpeed(self.ego_vehicle_id, 25)
        traci.vehicle.subscribeContext(self.ego_vehicle_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 100)
        for _ in range(20):
            traci.simulationStep()

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.start_simulation()
        self.current_step = 0
        state = self._get_state()
        return np.array(state).reshape(1, self.state_size)

    def _get_state(self):
        ego_vehicle = self.ego_vehicle_id
        try:
            ego_lane = traci.vehicle.getLaneIndex(ego_vehicle)
            ego_speed = traci.vehicle.getSpeed(ego_vehicle)
            ego_pos = traci.vehicle.getLanePosition(ego_vehicle)
            ego_lane_id = traci.vehicle.getLaneID(ego_vehicle)
            lane_max_speed = traci.lane.getMaxSpeed(ego_lane_id)
            norm_speed = ego_speed / lane_max_speed
            leading_vehicle = [0, 100, 0]
            following_vehicle = [0, 100, 0]
            left_leading = [0, 100, 0]
            left_following = [0, 100, 0]
            right_leading = [0, 100, 0]
            right_following = [0, 100, 0]
            if traci.vehicle.getLeader(ego_vehicle) is not None:
                lead_id, lead_dist = traci.vehicle.getLeader(ego_vehicle)
                if lead_id:
                    leading_vehicle = [1, lead_dist, traci.vehicle.getSpeed(lead_id) - ego_speed]
            surrounding_vehicles = traci.vehicle.getContextSubscriptionResults(ego_vehicle)
            if surrounding_vehicles:
                for v_id, v_data in surrounding_vehicles.items():
                    if v_id != ego_vehicle:
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
        except:
            ego_lane = 1
            norm_speed = 0
        lane_one_hot = [0, 0, 0]
        if ego_lane < len(lane_one_hot):
            lane_one_hot[ego_lane] = 1
        state = [norm_speed, *lane_one_hot, *leading_vehicle, *following_vehicle,
                 *left_leading, *left_following, *right_leading, *right_following]
        return state

    def step(self, action):
        if self.ego_vehicle_id not in traci.vehicle.getIDList():
            return np.zeros((1, self.state_size)), -100, True, {}
        old_lane = traci.vehicle.getLaneIndex(self.ego_vehicle_id)
        collision = False
        try:
            if action == 1 and old_lane > 0:
                traci.vehicle.changeLane(self.ego_vehicle_id, old_lane - 1, 5)
            elif action == 2 and old_lane < 2:
                traci.vehicle.changeLane(self.ego_vehicle_id, old_lane + 1, 5)
            current_speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
            lane_id = traci.vehicle.getLaneID(self.ego_vehicle_id)
            desired_speed = min(traci.lane.getMaxSpeed(lane_id), current_speed + 1)
            traci.vehicle.setSpeed(self.ego_vehicle_id, desired_speed)
            for _ in range(5):
                traci.simulationStep()
                if self.ego_vehicle_id in traci.simulation.getCollidingVehiclesIDList():
                    collision = True
                    break
                if self.ego_vehicle_id not in traci.vehicle.getIDList():
                    collision = True
                    break
        except:
            collision = True
        next_state = np.array(self._get_state()).reshape(1, self.state_size) if self.ego_vehicle_id in traci.vehicle.getIDList() else np.zeros((1, self.state_size))
        new_lane = traci.vehicle.getLaneIndex(self.ego_vehicle_id) if self.ego_vehicle_id in traci.vehicle.getIDList() else old_lane
        reward = -150 if collision else (1 if old_lane != new_lane else 0.1)
        self.current_step += 1
        done = collision or self.current_step >= self.max_steps or (self.ego_vehicle_id in traci.vehicle.getIDList() and traci.vehicle.getLanePosition(self.ego_vehicle_id) > 900)
        if done and not collision and self.ego_vehicle_id in traci.vehicle.getIDList():
            reward += 10
        return next_state, reward, done, {}

    def close(self):
        if traci.isLoaded():
            traci.close()

# 训练函数
def train_lane_change_agent(episodes=1000):
    results_dir = f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    models_dir = f"{results_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    env = SUMOEnvironment("ego_vehicle", SUMO_CONFIG_PATH, SUMO_BINARY)
    agent = LaneChangeDQNAgent(env.state_size, env.action_size)

    start_time = time.time()
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss > 0:  # 只有当训练发生时才记录损失
                total_loss += loss
                loss_count += 1
            state = next_state
            total_reward += reward

        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        episode_rewards.append(total_reward)

        # 更新目标模型
        if (episode + 1) % 10 == 0:
            agent.update_target_model()

        # 保存模型
        if (episode + 1) % 20 == 0:
            agent.save(f"{models_dir}/model_{episode + 1}")

        # 计算时间和进度
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (episode + 1) * episodes
        remaining_time = estimated_total_time - elapsed_time
        progress = (episode + 1) / episodes
        bar = '#' * int(20 * progress) + '-' * (20 - int(20 * progress))

        # 精简日志输出
        print(f"回合 {episode + 1}/{episodes} | 奖励: {total_reward:.2f} | 平均损失: {avg_loss:.4f}")
        print(f"[{bar}] {progress * 100:.1f}% | 已用: {elapsed_time / 60:.2f} 分钟 | 剩余: {remaining_time / 60:.2f} 分钟")

    env.close()
    agent.save(f"{models_dir}/model_final")

    # 设置支持中文的字体（可选，若无中文字体则使用英文标题）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 支持中文
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    except:
        print("未找到支持中文的字体，使用英文标题")
        title = "Episode Rewards"
    else:
        title = "回合奖励"

    # 绘制奖励曲线
    plt.plot(episode_rewards)
    plt.title(title)
    plt.xlabel("Episode" if "未找到" in locals() else "回合")
    plt.ylabel("Reward" if "未找到" in locals() else "奖励")
    plt.savefig(f"{results_dir}/rewards.png")
    plt.close()

    # 测试保存的模型
    loaded_model = load_model(f"{models_dir}/model_final.keras")
    test_agent = LaneChangeDQNAgent(env.state_size, env.action_size)
    test_agent.model = loaded_model
    test_agent.epsilon = 0  # 设置为0以禁用探索

    # 测试一个回合
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = test_agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    print(f"测试回合奖励: {total_reward:.2f}")
    env.close()

    print(f"训练完成，结果保存在: {results_dir}")
    return agent, episode_rewards

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    print("开始训练...")
    agent, rewards = train_lane_change_agent(episodes=1000)
    print("训练完成！")