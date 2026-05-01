"""
科研级：PPO vs PID 卫星姿态控制完整实验
包含：
- PPO训练
- PID控制
- 扰动测试
- 三图对比（θ, ω, τ）
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ===================== 卫星模型 =====================
class Satellite:
    def __init__(self, I=1.0):
        self.I = I
        self.theta = 0.0
        self.omega = 0.0

    def set_state(self, theta_deg, omega_deg=0):
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)

    def step(self, torque, dt, disturbance=0.0):
        torque_total = torque + disturbance
        alpha = torque_total / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt

# ===================== 环境 =====================
class SatEnv:
    def __init__(self):
        self.dt = 0.01
        self.max_steps = 300
        self.sat = Satellite()
        self.step_count = 0

    def reset(self):
        angle = np.random.uniform(-60, 60)
        self.sat.set_state(angle, 0)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return np.array([
            self.sat.theta / np.pi,
            self.sat.omega / 10.0
        ], dtype=np.float32)

    def step(self, action):
        disturbance = np.random.normal(0, 0.01)
        torque = np.clip(action, -2, 2)
        self.sat.step(torque, self.dt, disturbance)
        self.step_count += 1

        theta = self.sat.theta
        omega = self.sat.omega

        reward = -(theta**2 + 0.1*omega**2 + 0.01*torque**2)

        done = self.step_count >= self.max_steps or abs(theta) < np.radians(1)
        return self.get_state(), reward, done, {}

# ===================== PPO =====================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(2,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,1)
        )
        self.log_std = nn.Parameter(torch.tensor([-1.0]))

        self.critic = nn.Sequential(
            nn.Linear(2,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        value = self.critic(state)
        return dist, value

class PPO:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r,d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0,R)
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
            advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)

            s1 = ratio * advantage
            s2 = torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)*advantage

            actor_loss = -torch.min(s1,s2).mean()
            critic_loss = (returns - values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5*critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# ===================== PID =====================
class PID:
    def __init__(self):
        self.Kp = 3
        self.Ki = 0.5
        self.Kd = 1
        self.integral = 0
        self.prev_error = 0
        self.dt = 0.01

    def control(self, theta):
        error = -theta
        self.integral += error*self.dt
        derivative = (error-self.prev_error)/self.dt
        self.prev_error = error
        return self.Kp*error + self.Ki*self.integral + self.Kd*derivative

# ===================== 训练 =====================
def train():
    env = SatEnv()
    agent = PPO()

    reward_hist = []

    for ep in range(300):
        state = env.reset()
        states, actions, rewards, dones, log_probs = [],[],[],[],[]
        total = 0

        while True:
            s = torch.tensor(state, dtype=torch.float32)
            dist,_ = agent.model(s)
            action = dist.sample()

            ns, r, d, _ = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(r)
            dones.append(d)
            log_probs.append(dist.log_prob(action).item())

            state = ns
            total += r

            if d:
                break

        returns = agent.compute_returns(rewards,dones)
        agent.update(states,actions,log_probs,returns)

        reward_hist.append(total)

        if (ep+1)%20==0:
            print(ep+1, total)

    return agent, reward_hist

# ===================== 评估 =====================
def evaluate_ppo(agent):
    env = SatEnv()
    env.sat.set_state(30,0)

    t, th, w, u = [],[],[],[]

    for i in range(500):
        s = env.get_state()
        s = torch.tensor(s, dtype=torch.float32)
        with torch.no_grad():
            dist,_ = agent.model(s)
            action = dist.mean.item()

        disturbance = np.random.normal(0,0.01)
        env.sat.step(action, env.dt, disturbance)

        t.append(i*env.dt)
        th.append(np.degrees(env.sat.theta))
        w.append(np.degrees(env.sat.omega))
        u.append(action)

    return np.array(t),np.array(th),np.array(w),np.array(u)


def evaluate_pid():
    sat = Satellite()
    sat.set_state(30,0)
    pid = PID()

    t, th, w, u = [],[],[],[]

    for i in range(500):
        torque = pid.control(sat.theta)
        disturbance = np.random.normal(0,0.01)
        sat.step(torque,0.01,disturbance)

        t.append(i*0.01)
        th.append(np.degrees(sat.theta))
        w.append(np.degrees(sat.omega))
        u.append(torque)

    return np.array(t),np.array(th),np.array(w),np.array(u)

# ===================== 主程序 =====================
if __name__ == "__main__":
    agent, rewards = train()

    plt.plot(rewards)
    plt.title("Training Reward")
    plt.show()

    t1,th1,w1,u1 = evaluate_ppo(agent)
    t2,th2,w2,u2 = evaluate_pid()

    fig,ax = plt.subplots(3,1,figsize=(10,8))

    ax[0].plot(t1,th1,label='PPO')
    ax[0].plot(t2,th2,'--',label='PID')
    ax[0].set_ylabel('Angle')
    ax[0].legend(); ax[0].grid()

    ax[1].plot(t1,w1,label='PPO')
    ax[1].plot(t2,w2,'--',label='PID')
    ax[1].set_ylabel('Omega')
    ax[1].legend(); ax[1].grid()

    ax[2].plot(t1,u1,label='PPO')
    ax[2].plot(t2,u2,'--',label='PID')
    ax[2].set_ylabel('Torque')
    ax[2].set_xlabel('Time')
    ax[2].legend(); ax[2].grid()

    plt.tight_layout()
    plt.show()
