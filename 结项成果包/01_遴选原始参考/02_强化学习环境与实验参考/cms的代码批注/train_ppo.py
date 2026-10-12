
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
from satellite_env import SatelliteEnv

# 创建训练环境和评估环境
train_env = SatelliteEnv(config={
    'init_angle_range': (-0.5, 0.5),   # 随机初始角度
    'max_steps': 500,
    'disturbance': True,               # 训练时加干扰提高鲁棒性
    'noise': True,
})

eval_env = SatelliteEnv(config={
    'init_angle_range': (-0.5, 0.5),
    'max_steps': 500,
    'disturbance': False,              # 评估时不加干扰
    'noise': False,
})

# 包装环境以兼容SB3
train_env = DummyVecEnv([lambda: train_env])
eval_env = DummyVecEnv([lambda: eval_env])

# 创建评估回调
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/results",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# 创建PPO模型
model = PPO(
    "MlpPolicy",           # 多层感知机策略网络
    train_env,
    verbose=1,
    learning_rate=3e-4,    # 学习率
    n_steps=2048,          # 每次更新的步数
    batch_size=64,         # 批次大小
    n_epochs=10,           # 每个批次的训练轮数
    gamma=0.99,            # 折扣因子
    gae_lambda=0.95,       # GAE参数
    clip_range=0.2,        # PPO裁剪范围
    ent_coef=0.01,         # 熵系数（鼓励探索）
    tensorboard_log="./tensorboard_logs/"
)

# 开始训练
print("开始训练PPO控制器...")
model.learn(
    total_timesteps=200_000,  # 20万步，约10-20分钟
    callback=eval_callback
)

# 保存最终模型
model.save("ppo_satellite_final")
print("训练完成！模型已保存为 ppo_satellite_final")