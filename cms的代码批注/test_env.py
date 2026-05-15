import numpy as np
from satellite_env import SatelliteEnv
env = SatelliteEnv()
obs, info = env.reset()
print(f"初始状态: θ={np.degrees(obs[0]):.2f}°, ω={np.degrees(obs[1]):.2f}°/s")

total_reward = 0
for _ in range(200):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        break

print(f"随机策略总奖励: {total_reward:.2f}")