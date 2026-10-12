# train_ppo.py
# ============================================
# PPO 训练脚本（改良版）
# 特点：
# 1. 更稳定的超参数
# 2. 测试函数更规范（reset方式）
# 3. 绘图更完整
# ============================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from sat_env import SatelliteAttitudeEnv


# ======================== 创建环境 ========================
env = SatelliteAttitudeEnv()

# 检查环境合法性
check_env(env)


# ======================== 创建模型 ========================
model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=1e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,

    gamma=0.98,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,

    verbose=1,
    tensorboard_log="./ppo_sat_tensorboard/"
)


# ======================== 训练 ========================
total_timesteps = 300_000
model.learn(total_timesteps=total_timesteps)

model.save("ppo_satellite_improved")
print("模型已保存：ppo_satellite_improved.zip")


# ======================== 测试函数 ========================
def test_model(model, env, init_angle=30.0):
    obs, _ = env.reset()

    # 强制设定初始状态（用于对比）
    env.sat.set_state(init_angle, 0.0)
    obs = env._get_obs()

    steps = 500
    dt = env.dt

    time = []
    angles = []
    omegas = []
    torques = []

    for i in range(steps):
        t = i * dt

        theta = env.sat.get_angle_deg()
        omega = env.sat.get_omega_deg()

        action, _ = model.predict(obs, deterministic=True)
        torque = 2.0 * np.tanh(action[0])

        # 记录
        time.append(t)
        angles.append(theta)
        omegas.append(omega)
        torques.append(torque)

        # 推进
        env.sat.apply_torque(torque, dt)
        obs = env._get_obs()

    return np.array(time), np.array(angles), np.array(omegas), np.array(torques)


# ======================== 测试并绘图 ========================
time, angles, omegas, torques = test_model(model, env, init_angle=30.0)

plt.figure(figsize=(10, 8))

# 姿态角
plt.subplot(3, 1, 1)
plt.plot(time, angles)
plt.ylabel("Angle (deg)")
plt.title("PPO Attitude Control (Improved)")
plt.grid()

# 角速度
plt.subplot(3, 1, 2)
plt.plot(time, omegas)
plt.ylabel("Omega (deg/s)")
plt.grid()

# 力矩
plt.subplot(3, 1, 3)
plt.plot(time, torques)
plt.ylabel("Torque (Nm)")
plt.xlabel("Time (s)")
plt.grid()

plt.tight_layout()
plt.show()   