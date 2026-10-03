"""
强化学习训练脚本（satellite_ai 改进版）

改进点汇总（均以 "ds:" 注释标注）：
  ds6: 固定随机种子，保证实验可复现
  ds7: 新增 SAC 算法（连续控制首选），PPO 保留作对比（呼应中期报告计划）
  ds8: 训练/评估环境分离（训练加干扰+域随机化，评估用确定性环境），
       用 EvalCallback 按评估指标自动保存最优模型
  ds9: 用 VecNormalize 归一化奖励，稳定价值函数学习
       （观测已在 sat_env 内部归一化，故 norm_obs=False）

用法：
  python train.py sac   # 用 SAC 训练
  python train.py ppo   # 用 PPO 训练
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from single_axis.sat_env import SatelliteAttitudeEnv

# ds6: 固定随机种子，保证结果可复现
SEED = 42
set_random_seed(SEED)

# ds7: 算法选择（SAC 为连续控制首选；PPO 保留作对比）
ALGO = sys.argv[1].lower() if len(sys.argv) > 1 else "sac"
assert ALGO in ("sac", "ppo"), "用法: python train.py [sac|ppo]"

TOTAL_TIMESTEPS = 100_000   # 单轴双积分器环境较简单，10 万步通常足够收敛
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 20
HERE = os.path.dirname(os.path.abspath(__file__))   # single_axis/
LOG_DIR = os.path.join(HERE, "output", "tensorboard")
BEST_MODEL_DIR = os.path.join(HERE, "output", "models")


def make_env(train=True):
    """构建环境。

    ds8: 训练/评估环境分离 ——
      训练环境开启干扰 + 域随机化，逼策略学得更鲁棒；
      评估环境用确定性条件，保证评估可复现、结果可对比。
    """
    if train:
        config = {
            'disturbance': True,
            'randomize': True,          # ds4: 训练时域随机化
            # ds3: 非零干扰，训练出抗扰动的鲁棒策略
            'dist_periodic_amp': 0.05,
            'dist_periodic_freq': 0.5,
            'dist_noise_std': 0.01,
        }
    else:
        config = {'disturbance': False, 'randomize': False}
    return SatelliteAttitudeEnv(config=config)


def build_model(env, algo, seed=SEED):
    """按算法创建模型。"""
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=seed,
    )
    if algo == "sac":
        # ds7: SAC 使用自动熵调节，对奖励尺度不敏感，连续控制更稳
        model = SAC(
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            **common,
        )
    else:
        model = PPO(
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common,
        )
    return model


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # ---- 训练环境（含干扰 + 域随机化） ----
    train_env = DummyVecEnv([lambda: make_env(train=True)])
    # ds9: 归一化奖励，稳定价值函数学习（观测已在环境内归一化，故 norm_obs=False）
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # ---- 评估环境（确定性，不加干扰/随机化） ----
    # ds8/ds9: 评估环境同样用 VecNormalize 包裹（training=False 不更新统计量），
    #          并设 norm_reward=False，这样 EvalCallback 报告的是真实（未归一化）奖励，
    #          便于不同算法/参数公平比较。SB3 会在评估时自动同步训练环境的归一化统计量，
    #          所以两个环境必须都用 VecNormalize 包裹，否则会断言失败。
    eval_env = DummyVecEnv([lambda: make_env(train=False)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    # ds8: EvalCallback 定期在确定性环境评估，并按平均奖励自动保存最优模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(BEST_MODEL_DIR, "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    model = build_model(train_env, ALGO)

    print(f"[ds] 开始训练 {ALGO.upper()}，总步数 {TOTAL_TIMESTEPS}，种子 {SEED}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    # 保存最终模型（含归一化统计量，供后续评估/继续训练使用）
    model.save(os.path.join(BEST_MODEL_DIR, f"{ALGO}_satellite_final"))
    train_env.save(os.path.join(BEST_MODEL_DIR, f"{ALGO}_vecnormalize.pkl"))
    print(f"[ds] 训练完成，模型已保存至 {BEST_MODEL_DIR}/{ALGO}_satellite_final.zip")
