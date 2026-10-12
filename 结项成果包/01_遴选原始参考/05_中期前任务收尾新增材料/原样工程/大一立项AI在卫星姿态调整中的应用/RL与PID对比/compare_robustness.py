"""
RL(PPO) vs PID 三轴姿态控制 —— 加干扰的鲁棒性对比。

统一用 三轴强化学习/satellite_env3d.py 的动力学作为唯一仿真器，对同一组初始姿态，
在"同一确定性正弦干扰"下分别跑 RL 策略和 PID 控制器，比较：
    - 纠正时间（误差进入并此后一直保持 < 1° 的时刻）
    - 稳态残差（末段 5 秒的平均总指向误差，反映持续干扰下的残余偏差/振荡）

为测"持续干扰下的稳态保持能力"，每条轨迹固定跑满 20s（不提前终止），
避免"提前成功"掩盖干扰的长期影响。

干扰形式沿用项目 pid/three_axis 的正弦干扰，并按幅值 A ∈ {0, 0.02, 0.05, 0.10} N·m
扫描，观察两种控制器随干扰增强的退化情况：
    d(t) = A · [sin(2π·0.2t), 0.6·sin(2π·0.3t), 0.8·sin(2π·0.25t)]
（A = 0.05 时即项目原干扰。）

公平性说明：RL 策略训练时用的是 ±0.005 N·m 的均匀随机干扰，因此本对比考察的是
RL 在训练分布之外的鲁棒性；PID 增益（Kp=3, Ki=0.5, Kd=1）本就是配合该 0.05 正弦
干扰选取的。

用法: cd RL与PID对比 && python compare_robustness.py
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
RL_DIR = os.path.join(HERE, "..", "三轴强化学习")
sys.path.insert(0, RL_DIR)

from satellite_env3d import SatelliteEnv3D, _euler_to_quat  # noqa: E402  # type: ignore[import-not-found]
from metrics import settling_time  # noqa: E402  # type: ignore[import-not-found]

OUT_DIR = os.path.join(HERE, "成果图")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RL_DIR, "ppo_satellite3d")

THRESHOLD_DEG = 1.0
MAX_STEPS = 2000
DT = 0.01

# 扫描的干扰幅值（N·m）
AMPLITUDES = [0.0, 0.02, 0.05, 0.10]

# 全部 29 个测试姿态（度，ZYX 欧拉角 roll/pitch/yaw），与无干扰对比完全一致
CASES = []
for a in range(0, 91, 15):                      # roll 0–90°
    CASES.append((f"roll {a}°", (a, 0, 0)))
for a in range(0, 91, 15):                      # pitch 0–90°
    CASES.append((f"pitch {a}°", (0, a, 0)))
for a in range(0, 91, 15):                      # yaw 0–90°
    CASES.append((f"yaw {a}°", (0, 0, a)))
for t in range(0, 61, 15):                      # 三轴对称 0/15/30/45/60
    CASES.append((f"三轴对称 {t}°", (t, t, t)))
CASES.append(("三轴耦合 A", (45, -30, 45)))
CASES.append(("三轴耦合 B", (50, 30, -40)))
CASES.append(("三轴耦合 C", (30, 60, 20)))


def sine_disturbance(amplitude):
    """确定性正弦干扰（与项目 pid/three_axis 同款形式，幅度可调）。"""
    def d(t):
        return amplitude * np.array([
            np.sin(2 * np.pi * 0.2 * t),
            0.6 * np.sin(2 * np.pi * 0.3 * t),
            0.8 * np.sin(2 * np.pi * 0.25 * t),
        ])
    return d


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


def residual_error(errors, window_s=5.0):
    """末段 window_s 秒的平均总指向误差（度），反映持续干扰下的稳态残差。"""
    window = int(window_s / DT)
    seg = errors[-window:] if len(errors) >= window else errors
    return float(np.mean(seg))


def run_case(env, controller, euler_deg, is_rl):
    """从指定三轴欧拉角（度）跑一条固定 20s 轨迹，返回 (times, total_error_deg)。

    固定跑满 MAX_STEPS 步（忽略提前终止），以观测持续干扰下的稳态保持能力。
    """
    q0 = _euler_to_quat(*np.radians(euler_deg))
    obs, _ = env.reset(options={'q': q0, 'omega': np.zeros(3)})
    if not is_rl:
        controller.reset()
    times, errors = [0.0], [float(np.degrees(np.linalg.norm(obs[:3])))]
    for step in range(1, MAX_STEPS + 1):
        if is_rl:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            action = controller.compute(obs[:3])
        obs, _, _, _, _ = env.step(action)
        times.append(step * DT)
        errors.append(float(np.degrees(np.linalg.norm(obs[:3]))))
    return np.array(times), np.array(errors)


def summarize(rows):
    """rows: [(name, e, init_err, settle, residual), ...] → 汇总统计。"""
    settles = [r[3] for r in rows if r[3] is not None]
    residuals = [r[4] for r in rows]
    n_fail = sum(1 for r in rows if r[3] is None)
    avg_settle = float(np.mean(settles)) if settles else float('nan')
    avg_residual = float(np.mean(residuals))
    return avg_settle, avg_residual, n_fail


def main():
    model = PPO.load(MODEL_PATH)
    pid = PID3D()

    # A -> {'rl': rows, 'pid': rows}
    all_rows = {}
    for A in AMPLITUDES:
        env = SatelliteEnv3D(max_steps=MAX_STEPS, enable_disturbance=False,
                             disturbance_func=sine_disturbance(A))
        rl_rows, pid_rows = [], []
        for name, e in CASES:
            t, err = run_case(env, model, e, True)
            rl_rows.append((name, e, err[0], settling_time(err, t, THRESHOLD_DEG), residual_error(err)))
            t, err = run_case(env, pid, e, False)
            pid_rows.append((name, e, err[0], settling_time(err, t, THRESHOLD_DEG), residual_error(err)))
        all_rows[A] = {'rl': rl_rows, 'pid': pid_rows}

    # CSV：每种幅值、每个姿态、RL/PID 的纠正时间与残差
    csv_path = os.path.join(OUT_DIR, "compare_robustness.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["干扰幅值(N·m)", "姿态", "roll(°)", "pitch(°)", "yaw(°)", "初始总误差(°)",
                    "RL纠正时间(s)", "PID纠正时间(s)", "RL残差(°)", "PID残差(°)"])
        for A in AMPLITUDES:
            rl_rows = all_rows[A]['rl']
            pid_rows = all_rows[A]['pid']
            for rl_r, pid_r in zip(rl_rows, pid_rows):
                name, e, init, rl_st, rl_res = rl_r
                _, _, _, pid_st, pid_res = pid_r
                w.writerow([A, name, e[0], e[1], e[2], f"{init:.4f}",
                            f"{rl_st:.4f}" if rl_st is not None else "",
                            f"{pid_st:.4f}" if pid_st is not None else "",
                            f"{rl_res:.4f}", f"{pid_res:.4f}"])

    # 控制台汇总表
    print(f"{'幅值(N·m)':<10} | {'RL纠正(s)':>9} | {'PID纠正(s)':>10} | {'RL残差(°)':>9} | {'PID残差(°)':>10} | RL未稳定 | PID未稳定")
    for A in AMPLITUDES:
        rl_sum = summarize(all_rows[A]['rl'])
        pid_sum = summarize(all_rows[A]['pid'])
        rl_s = f"{rl_sum[0]:.2f}" if not np.isnan(rl_sum[0]) else "—"
        pid_s = f"{pid_sum[0]:.2f}" if not np.isnan(pid_sum[0]) else "—"
        print(f"{A:<10.2f} | {rl_s:>9} | {pid_s:>10} | {rl_sum[1]:>9.3f} | {pid_sum[1]:>10.3f} | "
              f"{rl_sum[2]:>8}/{len(CASES)} | {pid_sum[2]:>9}/{len(CASES)}")

    # 图 1：纠正时间 vs 干扰幅值
    rl_settle = [summarize(all_rows[A]['rl'])[0] for A in AMPLITUDES]
    pid_settle = [summarize(all_rows[A]['pid'])[0] for A in AMPLITUDES]
    plt.figure(figsize=(8, 5))
    plt.plot(AMPLITUDES, rl_settle, 'o-', label="RL(PPO)")
    plt.plot(AMPLITUDES, pid_settle, 's-', label="PID")
    plt.xlabel("干扰幅值 A (N·m)")
    plt.ylabel("平均纠正时间 (s)")
    plt.title("RL vs PID：纠正时间随干扰幅值的变化")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "robustness_correction_time.png"), dpi=150)
    plt.close()

    # 图 2：稳态残差 vs 干扰幅值（鲁棒性核心指标）
    rl_res = [summarize(all_rows[A]['rl'])[1] for A in AMPLITUDES]
    pid_res = [summarize(all_rows[A]['pid'])[1] for A in AMPLITUDES]
    plt.figure(figsize=(8, 5))
    plt.plot(AMPLITUDES, rl_res, 'o-', label="RL(PPO)")
    plt.plot(AMPLITUDES, pid_res, 's-', label="PID")
    plt.xlabel("干扰幅值 A (N·m)")
    plt.ylabel("末段 5 秒平均残差 (°)")
    plt.title("RL vs PID：稳态残差随干扰幅值的变化")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "robustness_residual_error.png"), dpi=150)
    plt.close()

    # 图 3：代表姿态在 A=0.05 干扰下的纠正曲线
    A_show = 0.05
    env = SatelliteEnv3D(max_steps=MAX_STEPS, enable_disturbance=False,
                         disturbance_func=sine_disturbance(A_show))
    targets = [(0, 0, 60), (45, 45, 45), (50, 30, -40)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, target in zip(axes, targets):
        t_rl, e_rl = run_case(env, model, target, True)
        t_pid, e_pid = run_case(env, pid, target, False)
        ax.plot(t_rl, e_rl, label="RL(PPO)")
        ax.plot(t_pid, e_pid, label="PID")
        ax.axhline(THRESHOLD_DEG, color='gray', linestyle='--', label=f"阈值 {THRESHOLD_DEG}°")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("总指向误差 (°)")
        ax.set_title(f"姿态 {target}°（干扰 A={A_show} N·m）")
        ax.grid(True)
        ax.legend()
    fig.suptitle("RL vs PID 干扰下纠正曲线")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "robustness_curves_A0.05.png"), dpi=150)
    plt.close()

    print(f"\n图片与 CSV 已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
