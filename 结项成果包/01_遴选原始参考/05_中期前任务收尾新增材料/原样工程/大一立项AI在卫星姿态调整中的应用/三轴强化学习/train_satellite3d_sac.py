"""训练三轴卫星姿态控制 SAC 模型（环境、奖励、训练预算 80 万步均与 PPO 一致）。"""
import os
import glob
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from satellite_env3d import SatelliteEnv3D

HERE = os.path.dirname(os.path.abspath(__file__))

# 创建环境（三轴同时随机偏转，含轴间耦合），与 PPO 完全一致
env = SatelliteEnv3D(max_steps=500, dt=0.01)
eval_env = SatelliteEnv3D(max_steps=500, dt=0.01)

# 评估回调：周期性评估，监控收敛并保存最优模型
os.makedirs(os.path.join(HERE, "logs3d", "best_sac"), exist_ok=True)
os.makedirs(os.path.join(HERE, "logs3d", "results_sac"), exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(HERE, "logs3d", "best_sac"),
    log_path=os.path.join(HERE, "logs3d", "results_sac"),
    eval_freq=20000,
    deterministic=True,
    render=False
)

# 断点续训：每隔 5 万步保存一次模型，避免中途中断（如电脑睡眠）导致前功尽弃
CKPT_DIR = os.path.join(HERE, "logs3d", "checkpoints_sac")
os.makedirs(CKPT_DIR, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=CKPT_DIR,
    name_prefix="sac"
)

total_timesteps = 800_000

# 若存在历史 checkpoint，则从步数最大的那个续训（用 -1 表示）；否则从头训练。
# 想强制重新训练时，删掉 logs3d/checkpoints_sac/ 目录即可。
ckpts = glob.glob(os.path.join(CKPT_DIR, "sac_*_steps.zip"))


def _steps(p):
    return int(os.path.basename(p).split("_")[1])


resume_from = max(ckpts, key=_steps) if ckpts else None

if resume_from is not None:
    done = _steps(resume_from)
    print(f"检测到 checkpoint，从 {done} 步续训 → {total_timesteps} 步")
    # SAC：离策略最大熵算法，连续控制常用强基线；采用 SB3 默认超参
    model = SAC.load(resume_from, env=env, device="cuda",
                     tensorboard_log=os.path.join(HERE, "satellite_tensorboard3d_sac"))
    model.verbose = 1
    # 注意：SAC 的 replay buffer 未随模型保存，续训后经验池为空，会短暂下降再回升
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback],
                reset_num_timesteps=False)
else:
    # SAC：离策略最大熵算法，连续控制常用强基线。
    # ent_coef 固定为 0.05 而非 "auto"：自动熵系数在本任务上会迅速塌缩到 ~4e-4，
    # 探索过早消失，策略卡在 ~3° 的局部最优（多个随机种子上重复出现）。
    # learning_starts 显式设为 2 万步，保证 replay buffer 里先有足够的随机经验。
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.05,
        learning_starts=20000,
        tensorboard_log=os.path.join(HERE, "satellite_tensorboard3d_sac")
    )
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

model.save(os.path.join(HERE, "sac_satellite3d"))
print("模型已保存:", os.path.join(HERE, "sac_satellite3d.zip"))
