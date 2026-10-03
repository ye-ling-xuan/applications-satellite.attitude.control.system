"""
三轴强化学习训练脚本（satellite_ai 新增模块）

ds18: 三轴训练脚本 —— 复用单轴 train.py 成熟做法（ds6 种子、ds7 SAC、
      ds8 训练/评估分离 + EvalCallback、ds9 VecNormalize）。
ds22: 课程学习 —— CurriculumCallback 训练中把初始姿态范围从 20° 线性增大到 90°（由易到难）。
ds24: 加大训练规模 —— 100 万步 + 更大的策略网络 [256,256]，提升精细控制表达力。

产物写入 three_axis/output/models_v2/（新版单独目录，保留上一阶段线性奖励成果）。

用法（在 satellite_ai 目录下运行）：
  python three_axis/train3d.py sac   # SAC（默认）
  python three_axis/train3d.py ppo   # PPO 对比
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from three_axis.sat_env3d import Satellite3DEnv

SEED = 42
set_random_seed(SEED)

ALGO = sys.argv[1].lower() if len(sys.argv) > 1 else "sac"
assert ALGO in ("sac", "ppo"), "用法: python three_axis/train3d.py [sac|ppo]"

# ds24: 三轴问题更难，加大到 100 万步（上一阶段 30 万步仍不收敛）
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 20_000
N_EVAL_EPISODES = 20
HERE = os.path.dirname(os.path.abspath(__file__))   # three_axis/
LOG_DIR = os.path.join(HERE, "output", "tensorboard")
BEST_MODEL_DIR = os.path.join(HERE, "output", "models_v2")  # 新版独立目录，保留旧成果


class CurriculumCallback(BaseCallback):
    """ds22: 课程学习 —— 训练中把初始姿态范围从 start_deg 线性增大到 end_deg。"""

    def __init__(self, env, start_deg=20.0, end_deg=90.0, total_steps=1_000_000):
        super().__init__()
        self._env = env
        self._start = start_deg
        self._end = end_deg
        self._total = total_steps

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self._total)
        cur = self._start + (self._end - self._start) * progress
        self._env.set_init_range(cur)
        return True


def make_env(train=True):
    if train:
        config = {'disturbance': True, 'randomize': True}   # ds16 训练时开域随机化
    else:
        config = {'disturbance': False, 'randomize': False}
    return Satellite3DEnv(config=config)


def build_model(env, algo, seed=SEED):
    # ds24: 更大的策略网络 [256,256]，提升对精细控制的表达力
    common = dict(policy="MlpPolicy", env=env, verbose=1,
                  tensorboard_log=LOG_DIR, seed=seed,
                  policy_kwargs=dict(net_arch=[256, 256]))
    if algo == "sac":
        return SAC(learning_rate=3e-4, buffer_size=300_000, batch_size=256, tau=0.005,
                   gamma=0.99, train_freq=1, gradient_steps=1, ent_coef="auto", **common)
    return PPO(learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10,
               gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, **common)


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    base_train_env = make_env(train=True)   # ds22: 保留引用，供课程学习回调更新
    train_env = DummyVecEnv([lambda: base_train_env])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = DummyVecEnv([lambda: make_env(train=False)])
    # ds9: 评估环境同样用 VecNormalize 包裹（training=False、norm_reward=False）
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(BEST_MODEL_DIR, "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    # ds22: 课程学习回调（初始姿态范围从 20° 线性增大到 90°）
    curriculum = CurriculumCallback(base_train_env, start_deg=20.0, end_deg=90.0,
                                    total_steps=TOTAL_TIMESTEPS)

    model = build_model(train_env, ALGO)
    print(f"[ds] 开始训练三轴 {ALGO.upper()}（课程学习 + shaping），"
          f"总步数 {TOTAL_TIMESTEPS}，种子 {SEED}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[curriculum, eval_callback])

    model.save(os.path.join(BEST_MODEL_DIR, f"{ALGO}_satellite3d_v2_final"))
    train_env.save(os.path.join(BEST_MODEL_DIR, f"{ALGO}_3d_v2_vecnormalize.pkl"))
    print(f"[ds] 三轴训练完成，模型已保存至 {BEST_MODEL_DIR}/{ALGO}_satellite3d_v2_final.zip")
