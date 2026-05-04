from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from satellite_env import SatelliteEnv   # 你的环境

# 创建训练环境
train_env = SatelliteEnv(max_steps=500, dt=0.01)

# 创建评估环境（使用相同参数，但独立实例）
eval_env = SatelliteEnv(max_steps=500, dt=0.01)

# 设置评估回调：每 10000 步评估一次，保存最佳模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# 创建 PPO 模型
model = PPO(
    "MlpPolicy",          # 多层感知机策略网络
    train_env,
    verbose=1,            # 打印训练日志
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # 鼓励探索
    tensorboard_log="./satellite_tensorboard/"
)

# 开始训练（总时间步数可以先设 100000 快速测试）
model.learn(total_timesteps=200000, callback=eval_callback)

# 保存最终模型
model.save("ppo_satellite_final")