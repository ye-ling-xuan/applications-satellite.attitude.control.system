"""
单轴卫星姿态强化学习训练脚本（修复版）
DDPG 算法，稳定训练，可与 PID 对比。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

# ======================== 1. 卫星模型 ========================
class Satellite:
    def __init__(self, I=1.0):
        self.I = I
        self.theta = 0.0   # rad
        self.omega = 0.0   # rad/s

    def set_state(self, theta_deg, omega_deg=0.0):
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)

    def apply_torque(self, torque, dt):
        alpha = torque / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt

    def get_angle_deg(self):
        return np.degrees(self.theta)

    def get_omega_deg(self):
        return np.degrees(self.omega)

# ======================== 2. PID 控制器 ========================
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target_deg, current_deg):
        target = np.radians(target_deg)
        current = np.radians(current_deg)
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = error
        return output

# ======================== 3. RL 环境（修复奖励缩放与状态归一化） ========================
class SatEnv:
    def __init__(self, dt=0.01, max_steps=1000, init_angle_range=[20.0, 40.0]):
        self.dt = dt
        self.max_steps = max_steps
        self.init_angle_range = init_angle_range
        self.sat = Satellite(I=1.0)
        self.action_bound = 2.0
        self.current_step = 0

    def reset(self):
        angle = np.random.uniform(*self.init_angle_range)
        self.sat.set_state(angle, 0.0)
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # 状态归一化：角度 / π，角速度 / 10.0
        theta_norm = self.sat.theta / np.pi
        omega_norm = self.sat.omega / 10.0
        return np.array([theta_norm, omega_norm], dtype=np.float32)

    def step(self, action):
        torque = np.clip(action[0], -1.0, 1.0) * self.action_bound
        self.sat.apply_torque(torque, self.dt)
        self.current_step += 1

        theta = self.sat.theta
        omega = self.sat.omega
        error = -theta   # 目标为0，误差 = 0 - theta
        raw_reward = - (10.0 * error**2 + 0.5 * omega**2 + 0.1 * torque**2)
        # 奖励缩放：除以10，防止值过大
        reward = raw_reward / 10.0

        done = (self.current_step >= self.max_steps)
        return self._get_state(), reward, done, {}

# ======================== 4. 神经网络 ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# ======================== 5. 经验回放池 ========================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32).reshape(-1, 1),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)

# ======================== 6. DDPG 智能体（修复版） ========================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-4,
                 gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau

        # 探索噪声参数
        self.noise_std = 0.3
        self.noise_decay = 0.99995
        self.min_noise = 0.05

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()

        if add_noise:
            action += np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ---- 更新 Critic ----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---- 更新 Actor ----
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ---- 软更新目标网络 ----
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 噪声衰减（带下限）
        self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)

# ======================== 7. 训练主循环 ========================
def train():
    env = SatEnv()
    agent = DDPGAgent(state_dim=2, action_dim=1)
    replay_buffer = ReplayBuffer()

    episodes = 200
    max_steps = env.max_steps
    batch_size = 64

    total_rewards = []

    # 预热阶段：纯随机动作填充经验池
    print("预热中，用随机动作填充经验池...")
    state = env.reset()
    for _ in range(1000):
        action = np.random.uniform(-1, 1, size=(1,))
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
    print("预热完成，开始训练。")

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

            agent.update(replay_buffer, batch_size)

            if done:
                break

        total_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{episodes}, Reward: {ep_reward:.2f}, Noise std: {agent.noise_std:.4f}")

    print("训练完成！")
    return agent, total_rewards

# ======================== 8. 评估与对比 ========================
def evaluate(agent, init_angle=30.0):
    env = SatEnv()
    env.sat.set_state(init_angle, 0.0)
    dt = env.dt
    steps = 1000
    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)

    for i in range(steps):
        time[i] = i * dt
        angles[i] = env.sat.get_angle_deg()
        omegas[i] = env.sat.get_omega_deg()
        # 获取归一化状态
        state = np.array([env.sat.theta / np.pi, env.sat.omega / 10.0], dtype=np.float32)
        action = agent.select_action(state, add_noise=False)
        torque = np.clip(action[0], -1.0, 1.0) * env.action_bound
        torques[i] = torque
        env.sat.apply_torque(torque, dt)

    return time, angles, omegas, torques

def evaluate_pid(init_angle=30.0):
    sat = Satellite()
    sat.set_state(init_angle, 0.0)
    pid = PIDController(Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01)

    dt = 0.01
    steps = 1000
    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)

    for i in range(steps):
        time[i] = i * dt
        angles[i] = sat.get_angle_deg()
        omegas[i] = sat.get_omega_deg()
        torque = pid.compute(0.0, angles[i])
        torques[i] = torque
        sat.apply_torque(torque, dt)

    return time, angles, omegas, torques

# ======================== 9. 运行 ========================
if __name__ == "__main__":
    # 训练
    agent, rewards = train()

    # 奖励曲线
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

    # 评估与对比
    t_rl, a_rl, w_rl, trq_rl = evaluate(agent, 30.0)
    t_pid, a_pid, w_pid, trq_pid = evaluate_pid(30.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(t_rl, a_rl, label='RL (DDPG)')
    axes[0].plot(t_pid, a_pid, label='PID', linestyle='--')
    axes[0].set_ylabel('Angle (deg)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t_rl, w_rl, label='RL (DDPG)')
    axes[1].plot(t_pid, w_pid, label='PID', linestyle='--')
    axes[1].set_ylabel('Omega (deg/s)')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(t_rl, trq_rl, label='RL (DDPG)')
    axes[2].plot(t_pid, trq_pid, label='PID', linestyle='--')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.show()