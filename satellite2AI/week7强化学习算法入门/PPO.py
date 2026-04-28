from stable_baselines3 import PPO
import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1")

# 创建PPO智能体
model = PPO(
    "MlpPolicy",          # 使用多层感知机策略网络
    env,
    verbose=1,            # 打印训练信息
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./cartpole_tensorboard/"
)

# 训练
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_cartpole")

# 加载并测试
del model
model = PPO.load("ppo_cartpole")

obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()