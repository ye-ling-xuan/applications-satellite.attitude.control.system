"""
评估与泛化测试脚本（satellite_ai 改进版）

ds11: 新增系统的 OOD（分布外）泛化评估 ——
      在训练范围之外的初始角度/角速度上测试策略，
      并统一用 metrics.py 输出量化指标，而非只画曲线。

用法：
  python evaluate.py <模型路径> [算法]
  例如：python evaluate.py satellite_ai_best_model/sac_satellite_final sac
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# ds12: 指定中文字体，避免绘图时中文/希腊字母显示为方框（Windows 下生效）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from stable_baselines3 import PPO, SAC

from single_axis.sat_env import SatelliteAttitudeEnv
from common.metrics import compute_metrics, print_metrics, print_table

ALGOS = {"sac": SAC, "ppo": PPO}


def load_model(path, algo="sac"):
    return ALGOS[algo].load(path)


def rollout(model, env, theta0_deg, omega0_deg=0.0):
    """
    在确定性环境（无干扰）下，从指定初始状态做一次完整 rollout，
    返回时间/角度/角速度/力矩序列。角度单位 deg，力矩单位 N·m。
    """
    # 先 reset 一次以初始化 np_random（即便 disturbance=False 也保持健壮）
    env.reset(seed=0)
    env.sat.set_state(np.radians(theta0_deg), np.radians(omega0_deg))
    env.step_count = 0
    env.hold_count = 0
    obs = env._get_obs()

    max_steps = env.config['max_steps']
    time = np.zeros(max_steps)
    angles = np.zeros(max_steps)
    omegas = np.zeros(max_steps)
    torques = np.zeros(max_steps)

    for i in range(max_steps):
        time[i] = i * env.config['dt']
        angles[i] = np.degrees(env.sat.theta)
        omegas[i] = np.degrees(env.sat.omega)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        torques[i] = info['torque']
        if terminated or truncated:
            return time[:i + 1], angles[:i + 1], omegas[:i + 1], torques[:i + 1]

    return time, angles, omegas, torques


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "models", "sac_satellite_final")
    algo = sys.argv[2].lower() if len(sys.argv) > 2 else "sac"

    if not os.path.exists(model_path + ".zip"):
        print(f"[ds] 未找到模型 {model_path}.zip，请先运行 train.py 训练")
        return

    model = load_model(model_path, algo)

    # ds11: OOD 测试集 —— 覆盖训练范围(20°~40°)之外的初始条件
    test_cases = [
        (10.0, 0.0),
        (30.0, 0.0),   # 训练范围内的参考点
        (60.0, 0.0),
        (90.0, 0.0),   # 训练范围之外的大角度
        (30.0, 10.0),  # 带非零初始角速度
    ]

    # 无干扰的确定性环境，仅评估控制器本身
    env = SatelliteAttitudeEnv(config={'disturbance': False, 'randomize': False})

    rows = []
    plt.figure(figsize=(10, 5))
    for theta0, omega0 in test_cases:
        time, angles, omegas, torques = rollout(model, env, theta0, omega0)
        m = compute_metrics(time, angles, omega=omegas, torque=torques, target=0.0)
        rows.append((f"{theta0:.0f}°/{omega0:.0f}°/s", m))
        plt.plot(time, angles, label=f"θ0={theta0:.0f}°, ω0={omega0:.0f}°/s")

    plt.axhline(0, color='gray', linestyle=':')
    plt.axhline(1, color='g', linestyle=':', alpha=0.5, label='±1° 稳定带')
    plt.axhline(-1, color='g', linestyle=':', alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(f"OOD 泛化测试（{algo.upper()}）")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    out_png = f"{algo}_ood_generalization.png"
    plt.savefig(out_png, dpi=150)
    print(f"[ds] 已保存曲线图 -> {out_png}")

    # ds11: 统一输出量化指标表
    print_table(rows)


if __name__ == "__main__":
    main()
