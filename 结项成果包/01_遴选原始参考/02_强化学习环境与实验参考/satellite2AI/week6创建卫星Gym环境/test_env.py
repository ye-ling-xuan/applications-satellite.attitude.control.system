from satellite2AI.week6创建卫星Gym环境.satellite_env import SatelliteEnv
import numpy as np

env = SatelliteEnv(max_steps=200)
obs, info = env.reset()
print("初始状态:", obs)

total_reward = 0
for i in range(200):
    # 随机动作
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if i % 20 == 0:
        env.render()
    if terminated or truncated:
        break
print(f"随机策略总奖励: {total_reward:.2f}")