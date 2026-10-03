"""
三轴续训脚本（satellite_ai 新增模块）

ds25: 近似续训 —— 因 SB3 的 model.save() 不保存回放缓冲区/优化器/VecNormalize 统计量，
      被截断的训练无法无缝续上；本脚本从 best_model.zip 加载策略，用新环境继续训练
      剩余步数（奖励归一化会重新累积，初期有价值函数适应期）。

用法（在 satellite_ai 目录下运行）：
  python three_axis/continue_train3d.py
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from three_axis.sat_env3d import Satellite3DEnv

SEED = 42
set_random_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))   # three_axis/
MODEL_PATH = os.path.join(HERE, "output", "models_v2", "best_model")
REMAINING = 91_126   # 1_000_000 - 908_874（上次被截断时已训练 908874 步）

base_env = Satellite3DEnv(config={'disturbance': True, 'randomize': True})
base_env.set_init_range(90.0)   # 课程学习已到最后阶段，初始范围固定在 ±90°
train_env = DummyVecEnv([lambda: base_env])
train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

model = SAC.load(MODEL_PATH, env=train_env)
print(f"[ds] 从 {MODEL_PATH} 继续训练 {REMAINING} 步（近似续训，种子 {SEED}）")
model.learn(total_timesteps=REMAINING, reset_num_timesteps=False)

model.save(os.path.join(HERE, "output", "models_v2", "sac_satellite3d_v2_final"))
train_env.save(os.path.join(HERE, "output", "models_v2", "sac_3d_v2_vecnormalize.pkl"))
print(f"[ds] 续训完成，模型已保存为 sac_satellite3d_v2_final.zip")
