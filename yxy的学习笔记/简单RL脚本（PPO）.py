"""
单轴卫星姿态控制（PPO版本，稳定训练
相比DDPG更稳定、更容易收敛
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ======================== 环境 ========================
class Satellite:
    def __init__(self, I=1.0):
        self.I = I
        self.theta = 0.0
        self.omega = 0.0

    def set_state(self, theta_deg, omega_deg=0.0):
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)

    def step(self, torque, dt):
        alpha = torque / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt

class SatEnv:
    def __init__(self):
        self.dt = 0.01
        self.max_steps = 300
        self.sat = Satellite()
        self.step_count = 0

    def reset(self):
        angle = np.random.uniform(-40, 40)
        self.sat.set_state(angle, 0)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return np.array([
            self.sat.theta / np.pi,
            self.sat.omega / 10.0
        ], dtype=np.float32)

    def step(self, action):
        torque = np.clip(action, -2, 2)
        self.sat.step(torque, self.dt)
        self.step_count += 1

        theta = self.sat.theta
        omega = self.sat.omega

        reward = -(theta**2 + 0.1*omega**2 + 0.01*torque**2)

        done = self.step_count >= self.max_steps or abs(theta) < np.radians(1)
        return self.get_state(), reward, done, {}

# ======================== PPO 网络 ========================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1))

        self.critic = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        value = self.critic(state)
        return dist, value

# ======================== PPO Agent ========================
class PPO:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, states, actions, log_probs, returns):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)

        for _ in range(10):
            dist, values = self.model(states)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            advantage = returns - values.detach().squeeze()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# ======================== 训练 ========================
def train():
    env = SatEnv()
    agent = PPO()

    rewards_history = []

    for ep in range(300):
        state = env.reset()

        states, actions, rewards, dones, log_probs = [], [], [], [], []

        total_reward = 0

        while True:
            state_t = torch.tensor(state, dtype=torch.float32)
            dist, _ = agent.model(state_t)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(dist.log_prob(action).item())

            state = next_state
            total_reward += reward

            if done:
                break

        returns = agent.compute_returns(rewards, dones)
        agent.update(states, actions, log_probs, returns)

        rewards_history.append(total_reward)

        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.2f}")

    return agent, rewards_history

# ======================== 主程序 ========================
if __name__ == "__main__":
    agent, rewards = train()

    plt.plot(rewards)
    plt.title("PPO Training")
    plt.show()
