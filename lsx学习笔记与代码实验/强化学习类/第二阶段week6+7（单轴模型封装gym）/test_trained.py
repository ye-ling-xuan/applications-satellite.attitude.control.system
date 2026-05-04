from stable_baselines3 import PPO
from satellite_env import SatelliteEnv
import numpy as np

model = PPO.load("ppo_satellite_final")
env = SatelliteEnv()

# 固定初始角度 30° 以便观察
env.sat.set_state(30.0, 0.0)
obs = env._get_obs()   # 注意：这里要获取归一化角度？但环境内部 step 会处理，我们直接重置后手动设置状态再重置？

# 更规范的做法：在 reset 后手动覆盖内部状态
obs, _ = env.reset()
env.sat.set_state(30.0, 0.0)
obs = np.array([env.sat.theta, env.sat.omega], dtype=np.float32)

done = False
total_reward = 0
step = 0

while not done and step < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()   # 打印当前状态
    done = terminated or truncated
    step += 1

print(f"总步数: {step}, 总奖励: {total_reward:.2f}")