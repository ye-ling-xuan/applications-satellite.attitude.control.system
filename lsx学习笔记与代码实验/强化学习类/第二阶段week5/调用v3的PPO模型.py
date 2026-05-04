from stable_baselines3 import PPO
import gymnasium as gym

# 加载保存好的模型
model = PPO.load("ppo_cartpole")

# 测试环境
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()