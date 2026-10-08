"""
RL(PPO) vs PID 三轴姿态控制对比。

统一用 rl/satellite_env3d.py 的动力学作为唯一仿真器，对同一组初始姿态
分别跑 RL 策略和 PID 控制器，比较纠正时间与稳态误差。
本版为「无干扰 / 无噪声 / 无死区」的干净对比（两边物理条件完全一致）。

用法: cd comparison && python compare_rl_pid.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.join(HERE, "..", "rl")
sys.path.insert(0, RL_DIR)

from satellite_env3d import SatelliteEnv3D, _euler_to_quat  # noqa: E402
from metrics import settling_time, steady_state_error  # noqa: E402

OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RL_DIR, "ppo_satellite3d")

# 测试姿态（度，ZYX 欧拉角 roll/pitch/yaw）
AXIS_SWEEPS = {
    "roll":  [(a, 0, 0) for a in range(0, 91, 15)],
    "pitch": [(0, a, 0) for a in range(0, 91, 15)],
    "yaw":   [(0, 0, a) for a in range(0, 91, 15)],
}
SYMMETRIC = [(t, t, t) for t in range(0, 61, 15)]          # 0,15,30,45,60
ASYMMETRIC = [(45, -30, 45), (50, 30, -40), (30, 60, 20)]

THRESHOLD_DEG = 1.0
MAX_STEPS = 2000
DT = 0.01


class PID3D:
    """三通道独立 PID，与项目 pid/three_axis/pid3d.py 同款参数（Kp=3, Ki=0.5, Kd=1）。"""

    def __init__(self, Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01,
                 output_limit=2.0, integral_limit=5.0):
        self.Kp = np.full(3, Kp, dtype=float)
        self.Ki = np.full(3, Ki, dtype=float)
        self.Kd = np.full(3, Kd, dtype=float)
        self.dt = dt
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def compute(self, error_vec):
        self.integral += error_vec * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error_vec - self.prev_error) / self.dt
        output = self.Kp * error_vec + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, -self.output_limit, self.output_limit)
        self.prev_error = error_vec.copy()
        return output


def run_episode(env, controller, euler_deg, is_rl):
    """从指定三轴欧拉角（度）跑一条轨迹，返回 (times, total_error_deg)。"""
    q0 = _euler_to_quat(*np.radians(euler_deg))
    obs, _ = env.reset(options={'q': q0, 'omega': np.zeros(3)})
    if not is_rl:
        controller.reset()
    times, errors = [], []
    done = False
    step = 0
    times.append(0.0)
    errors.append(float(np.degrees(np.linalg.norm(obs[:3]))))
    while not done and step < MAX_STEPS:
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            action = controller.compute(obs[:3])
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        times.append(step * DT)
        errors.append(float(np.degrees(np.linalg.norm(obs[:3]))))
    return np.array(times), np.array(errors)


def evaluate(controller, is_rl):
    """对全部测试姿态跑一遍，返回 rows（每项: cat, euler, init_err, settle, ss_err, times, errors）。"""
    env = SatelliteEnv3D(max_steps=MAX_STEPS, enable_disturbance=False)
    cases = []
    for cat, sweep in AXIS_SWEEPS.items():
        for e in sweep:
            cases.append((cat, e))
    for e in SYMMETRIC:
        cases.append(("三轴对称", e))
    for e in ASYMMETRIC:
        cases.append(("三轴耦合", e))
    rows = []
    for cat, e in cases:
        t, err = run_episode(env, controller, e, is_rl)
        st = settling_time(err, t, THRESHOLD_DEG)
        ss = steady_state_error(err, t, THRESHOLD_DEG)
        rows.append((cat, e, err[0], st, ss, t, err))
    return rows


def main():
    model = PPO.load(MODEL_PATH)
    pid = PID3D()

    rl_rows = evaluate(model, True)
    pid_rows = evaluate(pid, False)

    # 导出 CSV
    csv_path = os.path.join(OUT_DIR, "compare_rl_pid.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类型", "roll(°)", "pitch(°)", "yaw(°)", "初始总误差(°)",
                    "RL纠正时间(s)", "PID纠正时间(s)", "RL稳态误差(°)", "PID稳态误差(°)"])
        for (cat, e, init, rl_st, rl_ss, _, _), (_, _, _, pid_st, pid_ss, _, _) in zip(rl_rows, pid_rows):
            w.writerow([cat, e[0], e[1], e[2], f"{init:.4f}",
                        f"{rl_st:.4f}" if rl_st is not None else "",
                        f"{pid_st:.4f}" if pid_st is not None else "",
                        f"{rl_ss:.4f}" if not np.isnan(rl_ss) else "",
                        f"{pid_ss:.4f}" if not np.isnan(pid_ss) else ""])

    # 打印表格
    print(f"{'类型':<8} {'姿态':>18} | {'初始误差':>8} | {'RL纠正':>8} | {'PID纠正':>9} | {'RL稳态':>7} | {'PID稳态':>7}")
    for (cat, e, init, rl_st, rl_ss, _, _), (_, _, _, pid_st, pid_ss, _, _) in zip(rl_rows, pid_rows):
        rl_s = f"{rl_st:.2f}s" if rl_st is not None else "未稳定"
        pid_s = f"{pid_st:.2f}s" if pid_st is not None else "未稳定"
        rl_e = f"{rl_ss:.2f}" if not np.isnan(rl_ss) else "—"
        pid_e = f"{pid_ss:.2f}" if not np.isnan(pid_ss) else "—"
        print(f"{cat:<8} {str(e):>18} | {init:>7.1f}° | {rl_s:>8} | {pid_s:>9} | {rl_e:>7} | {pid_e:>7}")

    # 图 1：代表姿态纠正曲线对比
    targets = [(0, 0, 60), (45, 45, 45), (50, 30, -40)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, target in zip(axes, targets):
        rl_t = rl_e = pid_t = pid_e = None
        for cat, e, _, _, _, t, err in rl_rows:
            if e == target:
                rl_t, rl_e = t, err
        for cat, e, _, _, _, t, err in pid_rows:
            if e == target:
                pid_t, pid_e = t, err
        ax.plot(rl_t, rl_e, label="RL(PPO)")
        ax.plot(pid_t, pid_e, label="PID")
        ax.axhline(THRESHOLD_DEG, color='gray', linestyle='--', label=f"阈值 {THRESHOLD_DEG}°")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("总指向误差 (°)")
        ax.set_title(f"姿态 {target}°")
        ax.grid(True)
        ax.legend()
    fig.suptitle("RL vs PID 纠正曲线")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "compare_correction_curves.png"), dpi=150)
    plt.close(fig)

    # 图 2：纠正时间 vs 初始总误差
    plt.figure(figsize=(8, 5))
    xs = [r[2] for r in rl_rows]
    plt.plot(xs, [r[3] if r[3] is not None else float('nan') for r in rl_rows], 'o-', label="RL(PPO)")
    plt.plot(xs, [r[3] if r[3] is not None else float('nan') for r in pid_rows], 's-', label="PID")
    plt.xlabel("初始总指向误差 (°)")
    plt.ylabel("纠正时间 (s)")
    plt.title("RL vs PID 纠正时间")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_settling_time.png"), dpi=150)
    plt.close()

    # 图 3：稳态误差 vs 初始总误差
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r[4] if not np.isnan(r[4]) else float('nan') for r in rl_rows], 'o-', label="RL(PPO)")
    plt.plot(xs, [r[4] if not np.isnan(r[4]) else float('nan') for r in pid_rows], 's-', label="PID")
    plt.xlabel("初始总指向误差 (°)")
    plt.ylabel("稳态平均误差 (°)")
    plt.title("RL vs PID 稳态平均误差")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_steady_error.png"), dpi=150)
    plt.close()

    print(f"\n图片已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
