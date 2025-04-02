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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 解决 matplotlib 中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#####################
#     配置区域       #
#####################
class Config:
    sumo_binary = "sumo"       # 或 "sumo-gui"
    config_path = "a.sumocfg"
    ego_vehicle_id = "drl_ego_car"
    port_range = (8890, 8900)

    # BC 数据收集及训练参数
    bc_collect_episodes = 5
    bc_epochs = 10

    # PPO 训练回合和步数
    ppo_episodes = 200
    max_steps = 2000

    # PPO 超参数
    gamma = 0.99
    clip_epsilon = 0.2
    learning_rate = 3e-4
    batch_size = 256
    ppo_epochs = 3
    hidden_size = 256
    log_interval = 10

    # 状态/动作空间
    state_dim = 10
    action_dim = 3  # 0: 保持, 1: 左变, 2: 右变

    # 固定熵正则项（初期可较高，后期降低）
    entropy_coef = 0.03

    # 车道最高速度
    lane_max_speed = [33.33, 27.78, 22.22]

    # 奖励参数设置
    low_speed_threshold = 15.0
    low_speed_steps = 30
    low_speed_penalty = -2.0
    speed_increase_threshold = 2.0
    speed_increase_bonus = 0.4
    front_dist_improve_thresh = 5.0
    overtake_bonus = 0.5

    # GAE 参数
    gae_lambda = 0.95


#####################
#   SUMO 环境封装    #
#####################
class SumoEnv:
    def __init__(self):
        self.current_port = Config.port_range[0]
        self.sumo_process = None
        self.change_lane_count = 0
        self.collision_count = 0
        self.current_step = 0

        self.low_speed_count = 0
        self.prev_speed = 0.0
        self.prev_front_dist = 100.0

    def _init_sumo_cmd(self, port):
        return [
            Config.sumo_binary,
            "-c", Config.config_path,
            "--remote-port", str(port),
            "--no-warnings", "true",
            "--collision.action", "none",
            "--time-to-teleport", "-1",
            "--random"
        ]

    def reset(self):
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.change_lane_count = 0
        self.collision_count = 0
        self.current_step = 0
        self.low_speed_count = 0
        self.prev_speed = 0.0
        self.prev_front_dist = 100.0
        return self._get_state()

    def _start_sumo(self):
        for port in range(*Config.port_range):
            try:
                sumo_cmd = self._init_sumo_cmd(port)
                print(f"尝试连接SUMO，端口：{port}...")
                self.sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(2)
                traci.init(port)
                print(f"✅ SUMO连接成功，端口：{port}")
                self.current_port = port
                return
            except traci.exceptions.TraCIException:
                self._kill_sumo_processes()
                time.sleep(1)
        raise ConnectionError("无法在指定端口范围内连接SUMO！")

    def _add_ego_vehicle(self):
        if "ego_route" not in traci.route.getIDList():
            traci.route.add("ego_route", ["E0"])
        traci.vehicle.addFull(
            Config.ego_vehicle_id, "ego_route",
            typeID="car",
            depart="now",
            departLane="best",
            departSpeed="max"
        )
        for _ in range(20):
            traci.simulationStep()
            if Config.ego_vehicle_id in traci.vehicle.getIDList():
                return
        raise RuntimeError("自车生成失败！")

    def _get_state(self):
        state = np.zeros(Config.state_dim, dtype=np.float32)
        if Config.ego_vehicle_id not in traci.vehicle.getIDList():
            return state
        try:
            speed = traci.vehicle.getSpeed(Config.ego_vehicle_id)
            lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
            state[0] = speed / 33.33
            state[1] = lane / 2.0
            self._update_surrounding_vehicles(state)
            state[8] = state[1]
            state[9] = 1.0 if lane == 1 else 0.0
        except traci.TraCIException:
            pass
        return state

    def _update_surrounding_vehicles(self, state):
        ego_pos = traci.vehicle.getPosition(Config.ego_vehicle_id)
        ego_lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
        ranges = {
            'front': 100.0, 'back': 100.0,
            'left_front': 100.0, 'left_back': 100.0,
            'right_front': 100.0, 'right_back': 100.0
        }
        for veh_id in traci.vehicle.getIDList():
            if veh_id == Config.ego_vehicle_id:
                continue
            veh_lane = traci.vehicle.getLaneIndex(veh_id)
            veh_pos = traci.vehicle.getPosition(veh_id)
            dx = veh_pos[0] - ego_pos[0]
            dy = veh_pos[1] - ego_pos[1]
            distance = np.hypot(dx, dy)
            if veh_lane == ego_lane:
                if dx > 0:
                    ranges['front'] = min(ranges['front'], distance)
                else:
                    ranges['back'] = min(ranges['back'], distance)
            elif veh_lane == ego_lane - 1:
                if dx > 0:
                    ranges['left_front'] = min(ranges['left_front'], distance)
                else:
                    ranges['left_back'] = min(ranges['left_back'], distance)
            elif veh_lane == ego_lane + 1:
                if dx > 0:
                    ranges['right_front'] = min(ranges['right_front'], distance)
                else:
                    ranges['right_back'] = min(ranges['right_back'], distance)
        state[2] = ranges['front'] / 100.0
        state[3] = ranges['back'] / 100.0
        state[4] = ranges['left_front'] / 100.0
        state[5] = ranges['left_back'] / 100.0
        state[6] = ranges['right_front'] / 100.0
        state[7] = ranges['right_back'] / 100.0

    def step(self, action):
        done = False
        reward = 0.0
        try:
            lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
            if action == 1 and lane > 0:
                traci.vehicle.changeLane(Config.ego_vehicle_id, lane - 1, duration=2)
                self.change_lane_count += 1
            elif action == 2 and lane < 2:
                traci.vehicle.changeLane(Config.ego_vehicle_id, lane + 1, duration=2)
                self.change_lane_count += 1

            traci.simulationStep()
            reward = self._calculate_reward(action)
            self.current_step += 1
        except traci.TraCIException:
            done = True

        next_state = self._get_state()
        if traci.simulation.getTime() > 3600 or self.current_step >= Config.max_steps:
            done = True
        return next_state, reward, done

    def _calculate_reward(self, action):
        collisions = traci.simulation.getCollisions()
        if collisions:
            for collision in collisions:
                if collision.collider == Config.ego_vehicle_id or collision.victim == Config.ego_vehicle_id:
                    self.collision_count += 1
                    return -50.0

        speed = traci.vehicle.getSpeed(Config.ego_vehicle_id)
        lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
        lane_speed_limit = Config.lane_max_speed[lane]
        rel_speed = min(speed / lane_speed_limit, 1.0)
        speed_reward = rel_speed * 0.5

        if lane == 0:
            lane_reward = 0.2
        elif lane == 1:
            lane_reward = 0.1
        else:
            lane_reward = 0.0

        change_lane_bonus = 0.05 if action != 0 else 0.0
        ego_state = self._get_state()
        front_dist = ego_state[2] * 100
        front_penalty = -0.2 if front_dist < 10 else 0.0

        overtake_bonus = 0.0
        if front_dist > self.prev_front_dist + Config.front_dist_improve_thresh:
            overtake_bonus = Config.overtake_bonus
        self.prev_front_dist = front_dist

        if speed < Config.low_speed_threshold:
            self.low_speed_count += 1
        else:
            self.low_speed_count = 0
        low_speed_penalty = Config.low_speed_penalty if self.low_speed_count > Config.low_speed_steps else 0.0

        improved_speed_bonus = 0.0
        if speed > self.prev_speed + Config.speed_increase_threshold:
            improved_speed_bonus = Config.speed_increase_bonus
        self.prev_speed = speed

        total_reward = (speed_reward + lane_reward + change_lane_bonus +
                        front_penalty + overtake_bonus +
                        low_speed_penalty + improved_speed_bonus)
        return total_reward

    def _close(self):
        if self.sumo_process:
            try:
                traci.close()
            except traci.exceptions.TraCIException:
                pass
            finally:
                self.sumo_process.terminate()
                self.sumo_process.wait()
                self.sumo_process = None
                self._kill_sumo_processes()

    @staticmethod
    def _kill_sumo_processes():
        if os.name == 'nt':
            os.system("taskkill /f /im sumo.exe >nul 2>&1")
            os.system("taskkill /f /im sumo-gui.exe >nul 2>&1")


#####################
#   规则策略（Heuristic）用于收集 BC 数据
#####################
def rule_based_action(state, env):
    front_dist = state[2] * 100
    lane = int(state[1] * 2)
    can_left = (lane > 0)
    can_right = (lane < 2)
    if front_dist < 10:
        if can_left:
            return 1
        elif can_right:
            return 2
        else:
            return 0
    else:
        return 0


#####################
#   收集 BC 数据
#####################
def collect_bc_data(env, num_episodes=5):
    bc_data = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = rule_based_action(state, env)
            next_state, reward, done = env.step(action)
            bc_data.append((state, action))
            state = next_state
    return bc_data


#####################
#   BC Actor 网络
#####################
class BehaviorCloningNet(nn.Module):
    def __init__(self):
        super(BehaviorCloningNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(Config.state_dim, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.action_dim)
        )

    def forward(self, x):
        return self.actor(x)


def bc_train(bc_data, bc_epochs=10):
    net = BehaviorCloningNet()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    states = torch.FloatTensor([d[0] for d in bc_data])
    actions = torch.LongTensor([d[1] for d in bc_data])

    for epoch in range(bc_epochs):
        logits = net(states)
        loss = loss_fn(logits, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"[BC] Epoch {epoch+1}/{bc_epochs}, Loss={loss.item():.4f}")
    return net


#####################
#   PPO 网络
#####################
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(Config.state_dim, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(Config.state_dim, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class Agent:
    def __init__(self):
        self.policy = PPO()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.learning_rate)
        # 添加学习率调度器
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=10, verbose=True)
        self.memory = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []

    def load_bc_actor(self, bc_net):
        self.policy.actor.load_state_dict(bc_net.actor.state_dict())
        print("BC Actor权重已加载到PPO策略网络！")

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs, _ = self.policy(state_tensor)
        lane = int(state[1] * 2)
        mask = [1.0] * Config.action_dim
        if lane == 0:
            mask[1] = 0.0
        elif lane == 2:
            mask[2] = 0.0
        mask_tensor = torch.tensor(mask)
        probs = probs * mask_tensor
        probs = probs / (probs.sum() + 1e-8)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0:
            return

        states = torch.FloatTensor([m[0] for m in self.memory])
        actions = torch.LongTensor([m[1] for m in self.memory])
        old_log_probs = torch.FloatTensor([m[2] for m in self.memory])
        rewards = [m[3] for m in self.memory]

        # 计算折扣返回值与使用GAE计算优势
        returns = []
        advs = []
        R = 0
        gae = 0
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()
        for i in reversed(range(len(rewards))):
            R = rewards[i] + Config.gamma * R
            returns.insert(0, R)
            delta = rewards[i] + Config.gamma * (values[i+1] if i+1 < len(rewards) else 0) - values[i]
            gae = delta + Config.gamma * Config.gae_lambda * gae
            advs.insert(0, gae)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advs)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        for epoch in range(Config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, Config.batch_size):
                end = start + Config.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_probs, curr_values = self.policy(batch_states)
                lanes = (batch_states[:, 1] * 2).long()
                mask = torch.ones_like(new_probs)
                mask[lanes == 0, 1] = 0.0
                mask[lanes == 2, 2] = 0.0
                new_probs = new_probs * mask
                new_probs_sum = new_probs.sum(dim=1, keepdim=True)
                new_probs = new_probs / (new_probs_sum + 1e-8)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - Config.clip_epsilon, 1 + Config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(curr_values.squeeze(), batch_returns)
                entropy_bonus = dist.entropy().mean()
                loss = actor_loss - Config.entropy_coef * entropy_bonus + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.total_losses.append(loss.item())

        self.memory.clear()
        # 根据平均回合奖励调整学习率
        self.scheduler.step(returns.mean().item())



def main():
    # 第1步：收集BC数据
    env_bc = SumoEnv()
    bc_data = []
    print("开始使用规则策略收集BC数据...")
    for _ in range(Config.bc_collect_episodes):
        state = env_bc.reset()
        done = False
        while not done:
            action = rule_based_action(state, env_bc)
            next_state, reward, done = env_bc.step(action)
            bc_data.append((state, action))
            state = next_state
    env_bc._close()
    print(f"BC数据量：{len(bc_data)}")

    # 第2步：BC训练
    print("开始BC训练...")
    bc_net = bc_train(bc_data, bc_epochs=Config.bc_epochs)

    # 第3步：PPO训练（微调）
    env = SumoEnv()
    agent = Agent()
    agent.load_bc_actor(bc_net)

    best_reward = -float('inf')
    all_rewards = []
    lane_change_counts = []
    collision_counts = []
    total_steps_per_episode = []

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"bc_ppo_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    try:
        for episode in tqdm(range(1, Config.ppo_episodes + 1), desc="PPO训练回合"):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            while not done and step_count < Config.max_steps:
                action, log_prob = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.store((state, action, log_prob.item(), reward))
                state = next_state
                episode_reward += reward
                step_count += 1

            agent.update()

            all_rewards.append(episode_reward)
            lane_change_counts.append(env.change_lane_count)
            collision_counts.append(env.collision_count)
            total_steps_per_episode.append(step_count)

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(agent.policy.state_dict(), os.path.join(models_dir, "best_model.pth"))
                print(f"🎉 Episode {episode}, 新最佳模型！回合奖励：{best_reward:.2f}")

            if episode % Config.log_interval == 0:
                print(f"[Episode {episode}] Reward: {episode_reward:.2f}, Best: {best_reward:.2f}, "
                      f"LaneChange: {env.change_lane_count}, Collisions: {env.collision_count}")

    except KeyboardInterrupt:
        print("训练被手动中断...")
    finally:
        env._close()
        torch.save(agent.policy.state_dict(), os.path.join(models_dir, "last_model.pth"))

        # 绘制训练曲线图（保持原输出格式）
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        plt.plot(all_rewards)
        plt.title("回合奖励")

        plt.subplot(2, 2, 2)
        plt.plot(lane_change_counts)
        plt.title("变道次数")

        plt.subplot(2, 2, 3)
        plt.plot(collision_counts)
        plt.title("碰撞次数")

        plt.subplot(2, 2, 4)
        plt.plot(agent.total_losses)
        plt.title("训练损失")

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "training_curves.png"))
        plt.close()

        # 保存训练数据为 JSON 格式
        training_data = {
            "rewards": all_rewards,
            "lane_changes": lane_change_counts,
            "collisions": collision_counts,
            "steps": total_steps_per_episode,
            "actor_losses": agent.actor_losses,
            "critic_losses": agent.critic_losses,
            "total_losses": agent.total_losses
        }
        with open(os.path.join(results_dir, "training_data.json"), "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=4, ensure_ascii=False)

        print(f"训练完成，结果保存在目录: {results_dir}")


if __name__ == "__main__":
    if not (os.path.exists(Config.sumo_binary) or "SUMO_HOME" in os.environ):
        raise ValueError("SUMO路径错误，请检查SUMO是否正确安装并设置环境变量SUMO_HOME")
    if not os.path.exists(Config.config_path):
        raise ValueError(f"配置文件不存在: {Config.config_path}")
    main()
