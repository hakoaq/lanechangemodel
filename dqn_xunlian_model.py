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
from tensorflow.keras.layers import Dense
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


# DQN智能体（与PPO参数一致版本）
class LaneChangeDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.995  # 与PPO一致的折扣率
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # 与PPO一致的学习率
        self.batch_size = 256  # 与PPO一致的批量大小
        self.train_start = 1000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.update_count = 0
        self.update_frequency = 10

    def _build_model(self):
        # 使用与PPO相同的网络结构：3层512个神经元
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))  # 添加第三层，与PPO一致
        model.add(Dense(self.action_size, activation='linear'))
        
        # 使用固定学习率，不使用学习率调度器
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
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
            states[i] = state
            next_states[i] = next_state
        targets = self.model.predict(states, verbose=0)
        next_state_values = self.target_model.predict(next_states, verbose=0)
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            targets[i][action] = reward if done else reward + self.gamma * np.max(next_state_values[i])
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_count += 1
        if self.update_count % self.update_frequency == 0:
            self.update_target_model()

        return history.history['loss'][0]

    def save(self, name):
        if not name.endswith('.keras'):
            name += '.keras'
        self.model.save(name)


# SUMO环境（与PPO一致的奖励结构）
class SUMOEnvironment:
    def __init__(self, ego_vehicle_id, sumo_config_path, sumo_binary):
        self.ego_vehicle_id = ego_vehicle_id
        self.sumo_config_path = sumo_config_path
        self.sumo_binary = sumo_binary
        self.max_steps = 2000  # 增加步数，但不要像PPO那样只用2步，那可能是个错误
        self.current_step = 0
        self.sim_step_length = 0.1
        self.state_size = 10  # 与PPO一致的状态维度
        self.action_size = 3
        self.last_action = 0
        self.change_lane_count = 0  # 记录变道次数
        self.collision_count = 0    # 记录撞车次数

    def start_simulation(self):
        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config_path, "--start", "--step-length",
                    str(self.sim_step_length), "--no-warnings", "true",
                    "--collision.action", "none", "--time-to-teleport", "-1", "--random"]
        traci.start(sumo_cmd)
        traci.route.add("ego_route", ["E0"])
        traci.vehicle.add(self.ego_vehicle_id, "ego_route", typeID="car")
        traci.vehicle.setSpeedMode(self.ego_vehicle_id, 31)
        traci.vehicle.setLaneChangeMode(self.ego_vehicle_id, 0)
        traci.vehicle.moveTo(self.ego_vehicle_id, "E0_1", 5)  # 起始在中间车道
        traci.vehicle.setSpeed(self.ego_vehicle_id, 25)
        traci.vehicle.subscribeContext(self.ego_vehicle_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 100)
        for _ in range(20):
            traci.simulationStep()

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.start_simulation()
        self.current_step = 0
        self.last_action = 0
        self.change_lane_count = 0
        self.collision_count = 0
        return self._get_state()

    def _get_state(self):
        """获取车辆状态，与PPO使用相同的状态表示"""
        state = np.zeros(self.state_size, dtype=np.float32)
        try:
            # 基本状态
            speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
            lane = traci.vehicle.getLaneIndex(self.ego_vehicle_id)
            state[0] = speed / 33.33  # 归一化速度
            state[1] = lane / 2.0  # 归一化车道索引

            # 周围车辆信息
            ego_pos = traci.vehicle.getPosition(self.ego_vehicle_id)

            # 初始化距离值
            ranges = {
                'front': 100.0,
                'back': 100.0,
                'left_front': 100.0,
                'left_back': 100.0,
                'right_front': 100.0,
                'right_back': 100.0
            }

            # 检测周围车辆
            surrounding_vehicles = traci.vehicle.getContextSubscriptionResults(self.ego_vehicle_id)
            if surrounding_vehicles:
                for v_id, v_data in surrounding_vehicles.items():
                    if v_id != self.ego_vehicle_id:
                        v_lane = traci.vehicle.getLaneIndex(v_id)
                        v_pos = traci.vehicle.getPosition(v_id)
                        # 计算相对位置
                        dx = v_pos[0] - ego_pos[0]  # 纵向距离
                        dy = v_pos[1] - ego_pos[1]  # 横向距离
                        distance = np.hypot(dx, dy)

                        # 确定车辆相对位置
                        if v_lane == lane - 1:  # 左侧车道
                            if dx > 0:  # 前方
                                ranges['left_front'] = min(ranges['left_front'], distance)
                            else:  # 后方
                                ranges['left_back'] = min(ranges['left_back'], distance)
                        elif v_lane == lane + 1:  # 右侧车道
                            if dx > 0:  # 前方
                                ranges['right_front'] = min(ranges['right_front'], distance)
                            else:  # 后方
                                ranges['right_back'] = min(ranges['right_back'], distance)
                        elif v_lane == lane:  # 同车道
                            if dx > 0:  # 前方
                                ranges['front'] = min(ranges['front'], distance)
                            else:  # 后方
                                ranges['back'] = min(ranges['back'], distance)

            # 归一化距离并设置状态
            state[2] = ranges['front'] / 100.0
            state[3] = ranges['back'] / 100.0
            state[4] = ranges['left_front'] / 100.0
            state[5] = ranges['left_back'] / 100.0
            state[6] = ranges['right_front'] / 100.0
            state[7] = ranges['right_back'] / 100.0

            # 当前车道信息和目标车道信息（与PPO一致）
            state[8] = lane / 2.0  # 当前车道（归一化）
            state[9] = 1.0 if lane == 1 else 0.0  # 是否在中间车道

        except:
            pass

        return state.reshape(1, self.state_size)

    def step(self, action):
        if self.ego_vehicle_id not in traci.vehicle.getIDList():
            return np.zeros((1, self.state_size)), -10, True, {}  # 使用与PPO一致的碰撞惩罚

        old_lane = traci.vehicle.getLaneIndex(self.ego_vehicle_id)
        old_speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
        collision = False

        try:
            # 执行车道变更，使用duration=2与PPO一致
            if action == 1 and old_lane > 0:  # 左变道
                traci.vehicle.changeLane(self.ego_vehicle_id, old_lane - 1, 2)
                self.change_lane_count += 1
            elif action == 2 and old_lane < 2:  # 右变道
                traci.vehicle.changeLane(self.ego_vehicle_id, old_lane + 1, 2)
                self.change_lane_count += 1

            # 控制速度
            current_speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
            lane_id = traci.vehicle.getLaneID(self.ego_vehicle_id)
            desired_speed = min(traci.lane.getMaxSpeed(lane_id), current_speed + 1)
            traci.vehicle.setSpeed(self.ego_vehicle_id, desired_speed)

            # 只执行一步模拟，与PPO一致
            traci.simulationStep()

            # 检查碰撞
            if traci.simulation.getCollisions():
                for collision_data in traci.simulation.getCollisions():
                    if collision_data.collider == self.ego_vehicle_id or collision_data.victim == self.ego_vehicle_id:
                        collision = True
                        self.collision_count += 1
                        break
            
            if self.ego_vehicle_id not in traci.vehicle.getIDList():
                collision = True

        except:
            collision = True

        next_state = self._get_state()

        # 获取当前状态
        if self.ego_vehicle_id in traci.vehicle.getIDList():
            new_lane = traci.vehicle.getLaneIndex(self.ego_vehicle_id)
            new_speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
        else:
            new_lane = old_lane
            new_speed = 0

        # 计算奖励，使用与PPO一致的奖励结构
        reward = self._calculate_reward(action, old_lane, new_lane, new_speed, collision)

        self.current_step += 1
        done = collision or self.current_step >= self.max_steps or (
                self.ego_vehicle_id in traci.vehicle.getIDList() and
                traci.vehicle.getLanePosition(self.ego_vehicle_id) > 900
        )

        # 用于调试
        info = {
            'old_lane': old_lane,
            'new_lane': new_lane,
            'old_speed': old_speed,
            'new_speed': new_speed,
            'collision': collision,
            'step': self.current_step,
            'change_lane_count': self.change_lane_count,
            'collision_count': self.collision_count
        }

        self.last_action = action
        return next_state, reward, done, info

    def _calculate_reward(self, action, old_lane, new_lane, speed, collision):
        """使用与PPO一致的奖励计算方式"""
        if collision:
            return -10.0  # 碰撞惩罚，与PPO一致

        reward = 0.0

        # 速度奖励
        reward += (speed / 33.33) * 0.5  # 与PPO一致的速度奖励权重

        # 车道位置奖励（鼓励在中间车道）
        reward += (2 - abs(new_lane - 1)) * 0.3  # 与PPO一致的车道奖励权重

        # 变道奖励
        if action != 0:  # 如果尝试变道
            reward += 0.2  # 与PPO一致的变道奖励

        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()


# 训练函数
def train_lane_change_agent(episodes=1000):  # 改为20回合，与PPO一致
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"dqn_results_{timestamp}"
    models_dir = f"{results_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    env = SUMOEnvironment("ego_vehicle", SUMO_CONFIG_PATH, SUMO_BINARY)
    agent = LaneChangeDQNAgent(env.state_size, env.action_size)

    start_time = time.time()
    episode_rewards = []
    episode_steps = []
    episode_lane_changes = []
    train_info = {
        'loss': [],
        'epsilon': []
    }
    best_reward = -float('inf')  # 记录最佳奖励，与PPO一致

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state[0], action, reward, next_state[0], done)
            loss = agent.replay()

            if loss > 0:  # 只有当训练发生时才记录损失
                total_loss += loss
                loss_count += 1

            state = next_state
            total_reward += reward

        avg_loss = total_loss / max(1, loss_count)
        episode_rewards.append(total_reward)
        episode_steps.append(env.current_step)
        episode_lane_changes.append(env.change_lane_count)
        train_info['loss'].append(avg_loss)
        train_info['epsilon'].append(agent.epsilon)

        # 保存最佳模型，与PPO一致
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f"{models_dir}/best_model")
            print(f"🎉 发现新最佳模型！奖励：{best_reward:.2f}")

        # 每10回合保存一次模型，与PPO的log_interval一致
        if (episode + 1) % 10 == 0:
            agent.save(f"{models_dir}/dqn_model_{episode + 1}")
            print(f"Episode {episode + 1}, 奖励：{total_reward:.2f}, 最佳：{best_reward:.2f}, "
                  f"变道次数：{env.change_lane_count}, 撞车次数：{env.collision_count}")

        # 计算时间和进度
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (episode + 1) * episodes
        remaining_time = estimated_total_time - elapsed_time
        progress = (episode + 1) / episodes
        bar = '#' * int(20 * progress) + '-' * (20 - int(20 * progress))

        # 日志输出
        print(
            f"回合 {episode + 1}/{episodes} | 奖励: {total_reward:.2f} | 平均损失: {avg_loss:.4f} | 车道变更: {env.change_lane_count}")
        print(
            f"[{bar}] {progress * 100:.1f}% | 已用: {elapsed_time / 60:.2f} 分钟 | 剩余: {remaining_time / 60:.2f} 分钟")

    env.close()
    agent.save(f"{models_dir}/last_model")  # 保存最后的模型，与PPO一致

    # 绘制训练曲线
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title("回合奖励")
    plt.xlabel("回合")
    plt.ylabel("奖励")

    plt.subplot(2, 2, 2)
    plt.plot(episode_lane_changes)
    plt.title("车道变更次数")
    plt.xlabel("回合")
    plt.ylabel("变更次数")

    plt.subplot(2, 2, 3)
    plt.plot(train_info['loss'])
    plt.title("训练损失")
    plt.xlabel("回合")
    plt.ylabel("损失")

    plt.subplot(2, 2, 4)
    plt.plot(train_info['epsilon'])
    plt.title("探索率")
    plt.xlabel("回合")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png")
    plt.close()

    # 保存训练数据
    np.savez(f"{results_dir}/training_data.npz",
             rewards=episode_rewards,
             lane_changes=episode_lane_changes,
             steps=episode_steps,
             loss=train_info['loss'],
             epsilon=train_info['epsilon'])

    print(f"训练完成，结果保存在: {results_dir}")
    return agent, episode_rewards, episode_lane_changes


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    print("开始训练DQN模型...")
    agent, rewards, lane_changes = train_lane_change_agent(episodes=1000)  # 设置为20回合，与PPO一致
    print("训练完成！")
