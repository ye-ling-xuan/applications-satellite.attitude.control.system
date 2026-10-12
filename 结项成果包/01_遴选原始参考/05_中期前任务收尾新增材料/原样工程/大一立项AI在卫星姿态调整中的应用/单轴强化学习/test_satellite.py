from stable_baselines3 import PPO
from satellite_env import SatelliteEnv
import matplotlib.pyplot as plt
import numpy as np

# 加载模型
model = PPO.load("ppo_satellite_final")

# 创建环境（固定初始状态以便对比）
env = SatelliteEnv()
obs, info = env.reset()
# 可以手动设置初始角度，例如30度
env.theta = np.radians(30)
env.omega = 0.0
obs = np.array([env.theta, env.omega], dtype=np.float32)

done = False
total_reward = 0
steps = 0
history = {'time': [], 'angle': [], 'omega': [], 'torque': []}
dt = env.dt

while not done and steps < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    history['time'].append(steps * dt)
    history['angle'].append(np.degrees(obs[0]))
    history['omega'].append(np.degrees(obs[1]))
    history['torque'].append(action[0])
    
    total_reward += reward
    steps += 1

print(f"总步数: {steps}, 总奖励: {total_reward:.2f}")

# 绘制曲线
plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(history['time'], history['angle'])
plt.ylabel('Angle (deg)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(history['time'], history['omega'])
plt.ylabel('Omega (deg/s)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(history['time'], history['torque'])
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.grid(True)

plt.tight_layout()
plt.savefig('ai_control_result.png')
plt.show()