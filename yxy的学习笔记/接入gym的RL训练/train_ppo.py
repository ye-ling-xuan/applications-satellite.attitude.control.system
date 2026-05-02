# train_ppo.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sat_env import SatelliteAttitudeEnv

# 创建环境
env = SatelliteAttitudeEnv()

# 检查环境接口
check_env(env)

# 创建 PPO 智能体
model = PPO(
    "MlpPolicy",
    env,
    gamma=0.99,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
    tensorboard_log="./ppo_sat_tensorboard/"
)

# 训练
total_timesteps = 200_000
model.learn(total_timesteps=total_timesteps)
model.save("ppo_satellite")
print("模型已保存为 ppo_satellite.zip")

# ================== 测试与绘图 ==================
def test_model(model, env, init_angle=30.0):
    # 设置想要的初始状态
    env.sat.set_state(init_angle, 0.0)
    obs = np.array([env.sat.theta / np.pi, env.sat.omega / 10.0], dtype=np.float32)

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

        action, _ = model.predict(obs, deterministic=True)
        torque = np.clip(action[0], -2.0, 2.0)
        torques[i] = torque

        env.sat.apply_torque(torque, dt)
        obs = np.array([env.sat.theta / np.pi, env.sat.omega / 10.0], dtype=np.float32)

    return time, angles, omegas, torques

# 运行测试并画图
time, angles, omegas, torques = test_model(model, env, init_angle=30.0)

plt.figure(figsize=(8, 5))
plt.plot(time, angles)
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.title('PPO Controlled Satellite Attitude')
plt.grid(True)
plt.show()