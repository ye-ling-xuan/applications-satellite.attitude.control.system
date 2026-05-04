from satellite_env import SatelliteEnv
import numpy as np

env = SatelliteEnv()
obs, info = env.reset()
print("初始观测:", obs)

total_reward = 0
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"回合结束，步数: {env.step_count}, 总奖励: {total_reward:.2f}")
env.close()