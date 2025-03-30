import os
import sys
import time
import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime


# ========== 配置区 ==========
class Config:
    sumo_binary = "sumo"  # 如果SUMO_HOME已设置，直接使用"sumo"
    config_path = r"C:\Users\HaKoAq\Desktop\bishe\a.sumocfg"  # 确保路径正确
    ego_vehicle_id = "drl_ego_car"
    port_range = (8873, 8900)

    episodes = 1000
    max_steps = 2000  # 增加最大步数以延长训练时间，而不是只有2步
    gamma = 0.995
    clip_epsilon = 0.2
    learning_rate = 1e-3
    batch_size = 256
    hidden_size = 512
    save_dir = "./models"
    log_interval = 10

    state_dim = 10  # [速度, 车道, 前车距, 后车距, 左前, 左后, 右前, 右后, 当前车道, 目标车道]
    action_dim = 3  # 0:保持, 1:左变, 2:右变


# ========== SUMO环境封装 ==========
class SumoEnv:
    def __init__(self):
        self.current_port = Config.port_range[0]
        self.sumo_process = None
        self.change_lane_count = 0  # 记录变道次数
        self.collision_count = 0  # 记录撞车次数
        self.current_step = 0  # 添加当前步数计数

    def _init_sumo_cmd(self, port):
        return [
            Config.sumo_binary,
            "-c", Config.config_path,
            "--remote-port", str(port),
            "--no-warnings", "true",
            "--collision.action", "none",
            "--time-to-teleport", "-1",
            "--random",
        ]

    def reset(self):
        self._close()
        self._start_sumo()
        self._add_ego_vehicle()
        self.change_lane_count = 0
        self.collision_count = 0
        self.current_step = 0  # 重置步数计数
        return self._get_state()

    def _start_sumo(self):
        for port in range(*Config.port_range):
            try:
                sumo_cmd = self._init_sumo_cmd(port)
                print(f"尝试连接到SUMO，端口：{port}...")
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
        raise ConnectionError("无法在指定范围内连接到SUMO服务！")

    def _add_ego_vehicle(self):
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
            state[0] = speed / 33.33  # 速度归一化到 [0, 1]
            state[1] = lane / 2.0  # 车道归一化到 [0, 1]
            self._update_surrounding_vehicles(state)
            state[8] = state[1]
            state[9] = 1.0 if lane == 1 else 0.0  # 目标车道（中间车道）
        except traci.TraCIException:
            pass
        return state

    def _update_surrounding_vehicles(self, state):
        try:
            ego_pos = traci.vehicle.getPosition(Config.ego_vehicle_id)
            ego_lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
            ranges = {
                'front': (100.0, -1), 'back': (100.0, -1),
                'left_front': (100.0, -1), 'left_back': (100.0, -1),
                'right_front': (100.0, -1), 'right_back': (100.0, -1)
            }
            for veh_id in traci.vehicle.getIDList():
                if veh_id == Config.ego_vehicle_id:
                    continue
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                dx = traci.vehicle.getPosition(veh_id)[0] - ego_pos[0]
                dy = traci.vehicle.getPosition(veh_id)[1] - ego_pos[1]
                distance = np.hypot(dx, dy)
                if veh_lane == ego_lane - 1:
                    key = 'left_front' if dx > 0 else 'left_back'
                elif veh_lane == ego_lane + 1:
                    key = 'right_front' if dx > 0 else 'right_back'
                elif veh_lane == ego_lane:
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
        except traci.TraCIException:
            print("警告：更新周围车辆信息时出错。")

    def step(self, action):
        reward = 0.0
        done = False
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
            self.current_step += 1  # 增加步数计数
        except traci.TraCIException:
            done = True
        next_state = self._get_state()
        done = done or traci.simulation.getTime() > 3600 or self.current_step >= Config.max_steps
        return next_state, reward, done

    def _calculate_reward(self, action):
        reward = 0.0
        if traci.simulation.getCollisions():
            for collision in traci.simulation.getCollisions():
                if collision.collider == Config.ego_vehicle_id or collision.victim == Config.ego_vehicle_id:
                    self.collision_count += 1
                    return -10.0
        speed = traci.vehicle.getSpeed(Config.ego_vehicle_id)
        reward += (speed / 33.33) * 0.5
        lane = traci.vehicle.getLaneIndex(Config.ego_vehicle_id)
        reward += (2 - abs(lane - 1)) * 0.3
        if action != 0:
            reward += 0.2
        return reward

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
            nn.Linear(Config.hidden_size, Config.hidden_size),  # 增加一层
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(Config.state_dim, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),  # 增加一层
            nn.ReLU(),
            nn.Linear(Config.hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class Agent:
    def __init__(self):
        self.policy = PPO()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.learning_rate)
        self.memory = []
        # 添加损失跟踪
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs, _ = self.policy(state_tensor)
        lane = int(state[1] * 2)
        mask = [1.0] * Config.action_dim
        if lane == 0:
            mask[1] = 0.0
        elif lane == 2:
            mask[2] = 0.0
        probs *= torch.tensor(mask)
        probs /= probs.sum()
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self):
        if len(self.memory) < Config.batch_size:
            return {"actor_loss": 0, "critic_loss": 0, "total_loss": 0}

        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.LongTensor([t[1] for t in self.memory])
        old_log_probs = torch.FloatTensor([t[2] for t in self.memory])
        rewards = torch.FloatTensor([t[3] for t in self.memory])

        discounted_rewards = []
        running_reward = 0
        for r in reversed(rewards.numpy()):
            running_reward = r + Config.gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0

        for _ in range(5):
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

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_loss += loss.item()

        avg_actor_loss = total_actor_loss / 5
        avg_critic_loss = total_critic_loss / 5
        avg_total_loss = total_loss / 5

        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.total_losses.append(avg_total_loss)

        self.memory.clear()

        return {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "total_loss": avg_total_loss
        }


# ========== 训练主循环 ==========
def main():
    # 创建时间戳和目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"ppo_results_{timestamp}"
    models_dir = f"{results_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    env = SumoEnv()
    agent = Agent()
    best_reward = -float('inf')

    # 用于追踪训练过程
    episode_rewards = []
    episode_steps = []
    episode_lane_changes = []
    episode_collisions = []
    losses = []

    try:
        for episode in tqdm(range(1, Config.episodes + 1), desc="Training Episodes"):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < Config.max_steps:
                action, log_prob = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.memory.append((state, action, log_prob, reward))
                episode_reward += reward
                state = next_state
                step_count += 1

                if len(agent.memory) >= Config.batch_size:
                    loss_info = agent.update()
                    losses.append(loss_info["total_loss"])

            # 最后的更新
            if len(agent.memory) > 0:
                loss_info = agent.update()
                if loss_info["total_loss"] > 0:
                    losses.append(loss_info["total_loss"])

            # 追踪训练数据
            episode_rewards.append(episode_reward)
            episode_steps.append(env.current_step)
            episode_lane_changes.append(env.change_lane_count)
            episode_collisions.append(env.collision_count)

            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(agent.policy.state_dict(), f"{models_dir}/best_model.pth")
                print(f"🎉 发现新最佳模型！奖励：{best_reward:.2f}")

            # 输出日志
            if episode % Config.log_interval == 0:
                print(f"Episode {episode}, 奖励：{episode_reward:.2f}, 最佳：{best_reward:.2f}, "
                      f"变道次数：{env.change_lane_count}, 撞车次数：{env.collision_count}")

    except KeyboardInterrupt:
        print("手动中断")
    finally:
        env._close()
        # 保存最后的模型
        torch.save(agent.policy.state_dict(), os.path.join(models_dir, "last_model.pth"))

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
        plt.plot(losses)
        plt.title("训练损失")
        plt.xlabel("更新次数")
        plt.ylabel("损失")

        plt.subplot(2, 2, 4)
        plt.plot(episode_collisions)
        plt.title("碰撞次数")
        plt.xlabel("回合")
        plt.ylabel("次数")

        plt.tight_layout()
        plt.savefig(f"{results_dir}/training_curves.png")
        plt.close()

        # 保存训练数据为.npz文件
        np.savez(f"{results_dir}/training_data.npz",
                 rewards=episode_rewards,
                 lane_changes=episode_lane_changes,
                 steps=episode_steps,
                 collisions=episode_collisions,
                 actor_losses=agent.actor_losses,
                 critic_losses=agent.critic_losses,
                 total_losses=agent.total_losses)

        print(f"训练完成，结果保存在: {results_dir}")


if __name__ == "__main__":
    if not (os.path.exists(Config.sumo_binary) or "SUMO_HOME" in os.environ):
        raise ValueError(
            "SUMO路径错误，请检查SUMO是否正确安装并设置环境变量SUMO_HOME，或在Config类中设置正确的sumo_binary路径")
    if not os.path.exists(Config.config_path):
        raise ValueError(f"配置文件不存在：{Config.config_path}")
    main()
