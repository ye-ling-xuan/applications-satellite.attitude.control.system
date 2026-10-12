"""训练三轴卫星姿态控制 TD3 模型（环境、奖励、训练预算 80 万步均与 PPO 一致）。"""
import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from satellite_env3d import SatelliteEnv3D

HERE = os.path.dirname(os.path.abspath(__file__))

# 创建环境（三轴同时随机偏转，含轴间耦合），与 PPO 完全一致
env = SatelliteEnv3D(max_steps=500, dt=0.01)
eval_env = SatelliteEnv3D(max_steps=500, dt=0.01)

# 评估回调：周期性评估，监控收敛并保存最优模型
os.makedirs(os.path.join(HERE, "logs3d", "best_td3"), exist_ok=True)
os.makedirs(os.path.join(HERE, "logs3d", "results_td3"), exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(HERE, "logs3d", "best_td3"),
    log_path=os.path.join(HERE, "logs3d", "results_td3"),
    eval_freq=20000,
    deterministic=True,
    render=False
)

# TD3：离策略确定性算法（双 Critic + 延迟更新），连续控制强基线。
# action_noise 必须显式给出：TD3 的策略是确定性的，不加噪声就几乎没有探索，
# 只靠 learning_starts 的随机动作远远不够（早期版本漏了这一项，模型完全学不会）。
# learning_starts 显式设为 2 万步，保证 replay buffer 里先有足够的随机经验。
model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    learning_rate=1e-3,
    gamma=0.99,
    learning_starts=20000,
    action_noise=NormalActionNoise(mean=np.zeros(3), sigma=0.2 * np.ones(3)),
    tensorboard_log=os.path.join(HERE, "satellite_tensorboard3d_td3")
)

# 断点续训：每隔 5 万步保存一次模型，避免中途中断（如电脑睡眠）导致前功尽弃
os.makedirs(os.path.join(HERE, "logs3d", "checkpoints_td3"), exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=os.path.join(HERE, "logs3d", "checkpoints_td3"),
    name_prefix="td3"
)

total_timesteps = 800_000
model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

model.save(os.path.join(HERE, "td3_satellite3d"))
print("模型已保存:", os.path.join(HERE, "td3_satellite3d.zip"))
