import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from satellite_env3d import SatelliteEnv3D

HERE = os.path.dirname(os.path.abspath(__file__))

# 创建环境（三轴同时随机偏转，含轴间耦合）
env = SatelliteEnv3D(max_steps=500, dt=0.01)
eval_env = SatelliteEnv3D(max_steps=500, dt=0.01)

# 评估回调：周期性评估，监控收敛并保存最优模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(HERE, "logs3d", "best_model"),
    log_path=os.path.join(HERE, "logs3d", "results"),
    eval_freq=20000,
    deterministic=True,
    render=False
)

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

total_timesteps = 800_000
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

model.save(os.path.join(HERE, "ppo_satellite3d"))
print("模型已保存:", os.path.join(HERE, "ppo_satellite3d.zip"))
