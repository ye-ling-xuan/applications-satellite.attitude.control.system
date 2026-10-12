# train_ddpg.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from sat_env import SatelliteAttitudeEnv

env = SatelliteAttitudeEnv()
check_env(env)

model = DDPG(
    "MlpPolicy",
    env,
    gamma=0.99,
    tau=0.005,
    learning_rate=1e-4,
    batch_size=64,
    buffer_size=100000,
    verbose=1,
    tensorboard_log="./ddpg_sat_tensorboard/"
)

total_timesteps = 200_000
model.learn(total_timesteps=total_timesteps)
model.save("ddpg_satellite")
print("模型已保存为 ddpg_satellite.zip")

def test_model(model, env, init_angle=30.0):
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

time, angles, omegas, torques = test_model(model, env, 30.0)
plt.figure(figsize=(8, 5))
plt.plot(time, angles)
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.title('DDPG Controlled Satellite Attitude')
plt.grid(True)
plt.show()