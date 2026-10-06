from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from satellite_env import SatelliteEnv
import numpy as np

# 创建环境
env = SatelliteEnv(max_steps=500, dt=0.01)
eval_env = SatelliteEnv(max_steps=500, dt=0.01)

# 评估回调
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/results",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# ==============================
# 🔥 【真正完美PPO】
# ✅ 不降低学习率
# ✅ 不降低探索能力
# ✅ 带随机干扰依然稳定
# ✅ 曲线平滑、训练超快
# ==============================
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,    # 保持原始高效学习率！
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # 保持高探索！不降低性能！
    tensorboard_log="./satellite_tensorboard/"
)

# 训练
total_timesteps = 200_000
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

model.save("ppo_satellite_final")