import os
import sys
import time
import datetime
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import traci
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== 配置区 ==========
class Config:
    # SUMO配置
    sumo_binary = "sumo"  # 如果已设置SUMO_HOME，可直接使用"sumo"
    config_path = "a.sumocfg"
    ego_vehicle_id = "drl_ego_car"
    port_range = (8873, 8900)

    # 训练参数
    episodes = 1000
    max_steps = 2000
    gamma = 0.99
    clip_epsilon = 0.1
    learning_rate = 3e-4
    batch_size = 256
    ppo_epochs = 3       # 每次更新时的迭代轮数
    hidden_size = 512
    log_interval = 10

    # 状态和动作维度
    state_dim = 10
    action_dim = 3  # 0: 保持, 1: 左变, 2: 右变

# ========== SUMO环境封装 ==========
class SumoEnv:
    def __init__(self):
        self.current_port = Config.port_range[0]
        self.sumo_process = None
        self.change_lane_count = 0
        self.collision_count = 0
        self.current_step = 0

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
                print(f"端口{port}连接失败，尝试下一个端口...")
                self._kill_sumo_processes()
                time.sleep(1)
        raise ConnectionError("无法在指定端口范围内连接SUMO！")

    def _add_ego_vehicle(self):
        # 自定义一条ego_route，用于添加自车
        if "ego_route" not in traci.route.getIDList():
            traci.route.add("ego_route", ["E0"])
        traci.vehicle.addFull(
            Config.ego_vehicle_id, "ego_route",
            typeID="car", depart="now",
            departLane="best", departSpeed="max"
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
            state[8] = state[1]   # 当前车道
            # 目标车道（这里暂且设中间车道为目标，若当前即中间，则设1，否则0）
            state[9] = 1.0 if lane == 1 else 0.0
        except traci.TraCIException:
            pass
        return state

    def _update_surrounding_vehicles(self, state):
        ego_pos = traci.vehicle.getPosition(Config.ego_vehicle_id)
        ego_lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
        # 初始化各方向距离为较大值(单位米，后续要除100归一化)
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
                if dx > 0:  # 前车
                    if distance < ranges['front']:
                        ranges['front'] = distance
                else:       # 后车
                    if distance < ranges['back']:
                        ranges['back'] = distance
            elif veh_lane == ego_lane - 1:  # 左侧车道
                if dx > 0:
                    if distance < ranges['left_front']:
                        ranges['left_front'] = distance
                else:
                    if distance < ranges['left_back']:
                        ranges['left_back'] = distance
            elif veh_lane == ego_lane + 1:  # 右侧车道
                if dx > 0:
                    if distance < ranges['right_front']:
                        ranges['right_front'] = distance
                else:
                    if distance < ranges['right_back']:
                        ranges['right_back'] = distance
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
            # 自车可能已被移除(极端情况), 直接done
            done = True

        next_state = self._get_state()
        if traci.simulation.getTime() > 3600 or self.current_step >= Config.max_steps:
            done = True
        return next_state, reward, done

    def _calculate_reward(self, action):
        # 如果发生碰撞
        collisions = traci.simulation.getCollisions()
        if collisions:
            for collision in collisions:
                if (collision.collider == Config.ego_vehicle_id or 
                    collision.victim == Config.ego_vehicle_id):
                    self.collision_count += 1
                    return -50.0  # 碰撞惩罚

        # 根据自车速度和车道给予奖励
        speed = traci.vehicle.getSpeed(Config.ego_vehicle_id)
        speed_reward = (speed / 33.33) * 0.5
        lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
        lane_reward = (2 - abs(lane - 1)) * 0.3  # 假设中间车道优先

        # 对于变道给予少量奖励，鼓励尝试(可选)
        change_lane_bonus = 0.1 if action != 0 else 0.0

        # 可选：若与前车非常接近，也给一个负奖励，鼓励保持安全距离
        front_dist = traci.vehicle.getDistance(
            Config.ego_vehicle_id, traci.vehicle.getLaneID(Config.ego_vehicle_id), 10.0, 1
        )
        # 上面这个API只是示例，也可用 _update_surrounding_vehicles 里存的 front_dist
        # 这里简单写一个：当前车距离小于 5 米，给额外惩罚
        safe_distance_penalty = 0.0
        ego_state = self._get_state()
        front_dist_norm = ego_state[2] * 100  # front 距离的实际米数
        if front_dist_norm < 5.0:
            safe_distance_penalty = -1.0

        return speed_reward + lane_reward + change_lane_bonus + safe_distance_penalty

    def _close(self):
        if self.sumo_process:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
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

# ========== PPO算法实现 ==========
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(Config.state_dim, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
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
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# ========== Agent ==========
class Agent:
    def __init__(self):
        self.policy = PPO()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.learning_rate)
        self.memory = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs, _ = self.policy(state_tensor)
        # 动作屏蔽：不允许最左车道再左变、最右车道再右变
        lane = int(state[1] * 2)
        mask = [1.0] * Config.action_dim
        if lane == 0:
            mask[1] = 0.0
        elif lane == 2:
            mask[2] = 0.0
        probs = probs * torch.tensor(mask)
        probs = probs / probs.sum()  # 重新归一化
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store(self, transition):
        # transition = (state, action, log_prob, reward)
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < Config.batch_size:
            # 不足一个batch，先不更新
            return

        # 提取记忆
        states = torch.FloatTensor([m[0] for m in self.memory])
        actions = torch.LongTensor([m[1] for m in self.memory])
        old_log_probs = torch.FloatTensor([m[2] for m in self.memory])
        rewards = torch.FloatTensor([m[3] for m in self.memory])

        # 计算折扣回报
        discounted_rewards = []
        running = 0
        for r in reversed(rewards.numpy()):
            running = r + Config.gamma * running
            discounted_rewards.insert(0, running)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # 归一化
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        for _ in range(Config.ppo_epochs):
            new_probs, values = self.policy(states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages = discounted_rewards - values.squeeze().detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - Config.clip_epsilon, 1 + Config.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), discounted_rewards)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.total_losses.append(loss.item())

        self.memory.clear()

# ========== 训练主循环 ==========
def main():
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"ppo_results_{timestamp}"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    env = SumoEnv()
    agent = Agent()
    best_reward = -float('inf')

    # 记录数据
    all_rewards = []
    lane_change_counts = []
    collision_counts = []
    total_steps_per_episode = []

    try:
        for episode in tqdm(range(1, Config.episodes + 1), desc="训练回合"):
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

            # 回合结束后再统一更新(提高样本效率，减少噪声)
            agent.update()

            all_rewards.append(episode_reward)
            lane_change_counts.append(env.change_lane_count)
            collision_counts.append(env.collision_count)
            total_steps_per_episode.append(step_count)

            # 保存最佳模型
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
        # 保存最终模型
        torch.save(agent.policy.state_dict(), os.path.join(models_dir, "last_model.pth"))

        # 绘制曲线
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

        np.savez(os.path.join(results_dir, "training_data.npz"),
                 rewards=all_rewards,
                 lane_changes=lane_change_counts,
                 collisions=collision_counts,
                 steps=total_steps_per_episode,
                 actor_losses=agent.actor_losses,
                 critic_losses=agent.critic_losses,
                 total_losses=agent.total_losses)

        print(f"训练完成，结果保存在目录: {results_dir}")

if __name__ == "__main__":
    if not (os.path.exists(Config.sumo_binary) or "SUMO_HOME" in os.environ):
        raise ValueError("SUMO路径错误，请检查SUMO是否正确安装并设置环境变量SUMO_HOME")
    if not os.path.exists(Config.config_path):
        raise ValueError(f"配置文件不存在: {Config.config_path}")
    main()
