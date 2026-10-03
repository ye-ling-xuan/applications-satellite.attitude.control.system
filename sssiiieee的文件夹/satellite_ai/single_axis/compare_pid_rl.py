"""
单轴 PID vs 强化学习 系统量化对比（satellite_ai 新增模块）

ds26: 新增系统化对比 —— 在【同一动力学、同一初始条件、同一力矩限幅(2 N·m)】下，
      对比「网格整定出的最优 PID」与「训练好的 SAC」控制器，用统一 metrics 输出
      多指标（稳定时间/超调/IAE/ITAE/能耗/成功率），并分别给出【无干扰】与
      【含周期干扰】两种场景，作为结题的量化证据。

用法（在 satellite_ai 目录下运行）：
  python single_axis/compare_pid_rl.py
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
from common.metrics import compute_metrics, print_table

# 与单轴环境完全一致的物理参数，保证对比公平
I = 1.0
DT = 0.02
MAX_TORQUE = 2.0
THETA_SCALE = np.radians(90.0)
OMEGA_SCALE = np.radians(60.0)
HORIZON = 10.0   # 每次仿真时长 (s)


class PID:
    """单轴位置式 PID，与原始 controller.py 一致，含输出限幅（无抗饱和）。"""

    def __init__(self, Kp, Ki, Kd, dt=DT, output_limit=MAX_TORQUE):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        out = float(np.clip(self.Kp * error + self.Ki * self.integral
                            + self.Kd * derivative, -self.output_limit, self.output_limit))
        self.prev_error = error
        return out

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


def run_pid(pid, theta0_deg, omega0_deg=0.0, disturbance=False):
    sat = Satellite(I=I)
    sat.set_state(np.radians(theta0_deg), np.radians(omega0_deg))
    pid.reset()
    steps = int(HORIZON / DT)
    time = np.zeros(steps)
    angles = np.zeros(steps)
    torques = np.zeros(steps)
    for i in range(steps):
        t = i * DT
        time[i] = t
        angles[i] = np.degrees(sat.theta)
        torque = pid.compute(0.0, sat.theta)
        d = 0.05 * np.sin(2 * np.pi * 0.2 * t) if disturbance else 0.0
        torques[i] = torque
        sat.apply_torque(torque + d, DT)
    return time, angles, torques


def run_sac(model, theta0_deg, omega0_deg=0.0, disturbance=False):
    sat = Satellite(I=I)
    sat.set_state(np.radians(theta0_deg), np.radians(omega0_deg))
    steps = int(HORIZON / DT)
    time = np.zeros(steps)
    angles = np.zeros(steps)
    torques = np.zeros(steps)
    for i in range(steps):
        t = i * DT
        time[i] = t
        angles[i] = np.degrees(sat.theta)
        obs = np.array([sat.theta / THETA_SCALE, sat.omega / OMEGA_SCALE], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        torque = MAX_TORQUE * np.tanh(float(action[0]))
        d = 0.05 * np.sin(2 * np.pi * 0.2 * t) if disturbance else 0.0
        torques[i] = torque
        sat.apply_torque(torque + d, DT)
    return time, angles, torques


def _score(m):
    """PID 整定评分（越小越好）：稳定时间 + 超调 + 能耗 的加权和。"""
    s = m['settling_time']
    if not np.isfinite(s):
        s = HORIZON * 2.0
    return s + 0.1 * m['overshoot_deg'] + 0.1 * m['control_energy']


def tune_pid():
    """网格搜索最优 PID（在 30° 初始、无干扰条件下整定）。"""
    best = None
    for Kp in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for Ki in [0.0, 0.2, 0.5, 0.8]:
            for Kd in [0.0, 0.5, 1.0, 1.5, 2.0]:
                pid = PID(Kp, Ki, Kd)
                t, a, q = run_pid(pid, 30.0)
                s = _score(compute_metrics(t, a, torque=q))
                if best is None or s < best[0]:
                    best = (s, (Kp, Ki, Kd))
    print(f"[ds] 最优 PID: Kp={best[1][0]}, Ki={best[1][1]}, Kd={best[1][2]}  (评分 {best[0]:.3f})")
    return best[1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model = SAC.load(os.path.join(here, "output", "models", "sac_satellite_final"))

    Kp, Ki, Kd = tune_pid()
    pid = PID(Kp, Ki, Kd)
    test_angles = [10.0, 30.0, 45.0, 60.0, 90.0]

    for disturbance, title in [(False, "无干扰"), (True, "含周期干扰")]:
        rows = []
        plt.figure(figsize=(10, 5))
        for th in test_angles:
            tp, ap, qp = run_pid(pid, th, disturbance=disturbance)
            ts, as_, qs = run_sac(model, th, disturbance=disturbance)
            rows.append((f"PID {th:.0f}°", compute_metrics(tp, ap, torque=qp)))
            rows.append((f"SAC {th:.0f}°", compute_metrics(ts, as_, torque=qs)))
            plt.plot(tp, ap, 'b-', alpha=0.55, label="PID" if th == test_angles[0] else None)
            plt.plot(ts, as_, 'r--', alpha=0.55, label="SAC" if th == test_angles[0] else None)

        plt.axhline(0, color='gray', linestyle=':')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(f"单轴 PID vs SAC（{title}）")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        png = os.path.join(here, "output", f"pid_vs_sac_{'dist' if disturbance else 'nodist'}.png")
        plt.savefig(png, dpi=150)
        print(f"\n=== {title}场景对比 ===")
        print_table(rows)
        print(f"[ds] 已保存 -> {png}")


if __name__ == "__main__":
    main()
