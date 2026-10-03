"""
PID vs 强化学习 vs MPC 三方系统对比（satellite_ai 新增模块）

ds29: 新增三方对比 —— 在【同一动力学、同一初始条件、同一力矩限幅】下，
      对比「网格整定 PID」「训练好的 SAC」「基于辨识模型的 MPC」三种控制器，
      用统一 metrics 输出多指标，作为结题「经典控制 vs 智能控制 vs 预测控制」的证据。

用法（在 satellite_ai 目录下运行）：
  python mpc/compare.py
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from single_axis.sat_env import Satellite
from single_axis.compare_pid_rl import PID, run_pid, run_sac, tune_pid, I, DT, MAX_TORQUE
from common.metrics import compute_metrics, print_table
from mpc.system_id import generate_data, identify
from mpc.mpc_controller import MPCController

# ds28: MPC 权重（状态 x=[theta, omega]，单位弧度；u 单位 N·m）
MPC_Q = np.diag([100.0, 5.0])   # 角度误差权重 100，角速度权重 5
MPC_R = 0.5                      # 控制量权重
MPC_HORIZON = 20


def run_mpc(mpc, theta0_deg, omega0_deg=0.0):
    sat = Satellite(I=I)
    sat.set_state(np.radians(theta0_deg), np.radians(omega0_deg))
    steps = int(10.0 / DT)
    time = np.zeros(steps)
    angles = np.zeros(steps)
    torques = np.zeros(steps)
    for i in range(steps):
        time[i] = i * DT
        angles[i] = np.degrees(sat.theta)
        u = mpc.control(np.array([sat.theta, sat.omega]))
        torques[i] = u
        sat.apply_torque(u, DT)
    return time, angles, torques


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # 1. 系统辨识（ds27）
    X, U, Y = generate_data()
    A, B = identify(X, U, Y)

    # 2. 三种控制器
    Kp, Ki, Kd = tune_pid()                       # 网格整定 PID
    pid = PID(Kp, Ki, Kd)
    mpc = MPCController(A, B, MPC_Q, MPC_R, horizon=MPC_HORIZON, max_torque=MAX_TORQUE)
    model = SAC.load(os.path.join(os.path.dirname(here), "single_axis", "output",
                                  "models", "sac_satellite_final"))

    test_angles = [10.0, 30.0, 45.0, 60.0, 90.0]

    rows = []
    plt.figure(figsize=(10, 5))
    for th in test_angles:
        tp, ap, qp = run_pid(pid, th)
        tm, am, qm = run_mpc(mpc, th)
        ts, as_, qs = run_sac(model, th)
        rows.append((f"PID {th:.0f}°", compute_metrics(tp, ap, torque=qp)))
        rows.append((f"MPC {th:.0f}°", compute_metrics(tm, am, torque=qm)))
        rows.append((f"SAC {th:.0f}°", compute_metrics(ts, as_, torque=qs)))
        plt.plot(tp, ap, 'b-', alpha=0.5, label="PID" if th == test_angles[0] else None)
        plt.plot(tm, am, 'g-.', alpha=0.6, label="MPC" if th == test_angles[0] else None)
        plt.plot(ts, as_, 'r--', alpha=0.5, label="SAC" if th == test_angles[0] else None)

    plt.axhline(0, color='gray', linestyle=':')
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("PID vs MPC vs SAC（单轴）")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    png = os.path.join(here, "pid_mpc_sac.png")
    plt.savefig(png, dpi=150)
    print(f"[ds] 已保存三方对比图 -> {png}")
    print("\n=== PID vs MPC vs SAC 三方对比（无干扰） ===\n")
    print_table(rows)


if __name__ == "__main__":
    main()
