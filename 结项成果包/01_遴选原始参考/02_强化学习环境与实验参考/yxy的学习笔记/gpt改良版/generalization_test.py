# generalization_test.py
# ============================================
# PPO 泛化能力测试
# 测试不同初始角度下的控制效果
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sat_env import SatelliteAttitudeEnv


# ======================== 测试函数 ========================
def run_episode(model, env, init_angle):
    env.reset()
    env.sat.set_state(init_angle, 0.0)

    obs = env._get_obs()

    steps = 500
    dt = env.dt

    time = []
    angles = []

    for i in range(steps):
        t = i * dt
        theta = env.sat.get_angle_deg()

        action, _ = model.predict(obs, deterministic=True)
        torque = 2.0 * np.tanh(action[0])

        env.sat.apply_torque(torque, dt)
        obs = env._get_obs()

        time.append(t)
        angles.append(theta)

    return np.array(time), np.array(angles)


# ======================== 主程序 ========================
env = SatelliteAttitudeEnv()
model = PPO.load("ppo_satellite_improved")

# 测试多个初始角度
test_angles = [10, 20, 30, 45, 60]

plt.figure(figsize=(8, 5))

for angle in test_angles:
    time, angles = run_episode(model, env, angle)
    plt.plot(time, angles, label=f"{angle} deg")

plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("Generalization Test of PPO Controller")
plt.legend()
plt.grid()
plt.show()