import os
from stable_baselines3 import PPO
from satellite_env3d import SatelliteEnv3D

HERE = os.path.dirname(os.path.abspath(__file__))

# 创建环境
env = SatelliteEnv3D(max_steps=500, dt=0.01)

# 训练：三轴维度更高，预算比单轴大
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log=os.path.join(HERE, "satellite_tensorboard3d")
)

total_timesteps = 500_000
model.learn(total_timesteps=total_timesteps)

model.save(os.path.join(HERE, "ppo_satellite3d"))
print("模型已保存:", os.path.join(HERE, "ppo_satellite3d.zip"))
