"""
三轴 RL 评测：对单轴（yaw）0–90° 初始偏角扫描，测纠正时间与平均指向误差，出图并导出 CSV。
用法: cd rl && python evaluate_three_axis.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from satellite_env3d import SatelliteEnv3D, _euler_to_quat
from metrics import settling_time, mean_error, steady_state_error

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "ppo_satellite3d")
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

ANGLES_DEG = list(range(0, 91, 10))
THRESHOLD_DEG = 1.0
MAX_STEPS = 1000


def pointing_error_deg(obs):
    # obs 前 3 维是误差旋转向量，其模长即总指向误差角（rad）
    return float(np.degrees(np.linalg.norm(obs[:3])))


def run_episode(model, env, yaw_deg):
    """从指定 yaw 偏角跑一条确定性轨迹，返回 (times, errors_deg)。"""
    q0 = _euler_to_quat(0.0, 0.0, np.radians(yaw_deg))
    obs, _ = env.reset(options={'q': q0, 'omega': np.zeros(3)})
    times, errors_deg = [], []
    dt = env.dt
    done = False
    step = 0
    # 记录初始误差（t=0）
    times.append(0.0)
    errors_deg.append(pointing_error_deg(obs))
    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        times.append(step * dt)
        errors_deg.append(pointing_error_deg(obs))
    return np.array(times), np.array(errors_deg)


def main():
    model = PPO.load(MODEL_PATH)
    env = SatelliteEnv3D(max_steps=MAX_STEPS)

    settle, avg_err, ss_err, curves = {}, {}, {}, {}
    print(f"{'初始偏角':>8} | {'纠正时间':>10} | {'全程均误':>10} | {'稳态均误':>10}")
    for a in ANGLES_DEG:
        t, e = run_episode(model, env, a)
        settle[a] = settling_time(e, t, THRESHOLD_DEG)
        avg_err[a] = mean_error(e)
        ss_err[a] = steady_state_error(e, t, THRESHOLD_DEG)
        curves[a] = (t, e)
        s = f"{settle[a]:.2f}s" if settle[a] is not None else "未稳定"
        s2 = f"{ss_err[a]:.2f}°" if not np.isnan(ss_err[a]) else "——"
        print(f"{a:>7}° | {s:>10} | {avg_err[a]:>9.2f}° | {s2:>10}")

    # 导出结果表（UTF-8 CSV，Excel 可直接打开）
    csv_path = os.path.join(OUT_DIR, "three_axis_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["初始偏角(°)", "纠正时间(s)", "全程平均误差(°)", "稳态平均误差(°)"])
        for a in ANGLES_DEG:
            w.writerow([
                a,
                f"{settle[a]:.4f}" if settle[a] is not None else "",
                f"{avg_err[a]:.4f}",
                f"{ss_err[a]:.4f}" if not np.isnan(ss_err[a]) else "",
            ])

    # 图 1：代表角度纠正曲线
    plt.figure(figsize=(10, 5))
    for a in [10, 30, 60, 90]:
        t, e = curves[a]
        plt.plot(t, e, label=f"{a}°")
    plt.axhline(THRESHOLD_DEG, color='gray', linestyle='--', label=f"阈值 {THRESHOLD_DEG}°")
    plt.xlabel("时间 (s)")
    plt.ylabel("指向误差 (°)")
    plt.title("三轴 RL 纠正曲线（yaw 单轴偏转）")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "three_axis_correction_curves.png"), dpi=150)
    plt.close()

    # 图 2：纠正时间 vs 初始偏角
    plt.figure(figsize=(8, 5))
    ys = [settle[a] if settle[a] is not None else float('nan') for a in ANGLES_DEG]
    plt.plot(ANGLES_DEG, ys, marker='o')
    plt.xlabel("初始偏角 (°)")
    plt.ylabel("纠正时间 (s)")
    plt.title("三轴 RL 纠正时间 vs 初始偏角")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "three_axis_settling_time.png"), dpi=150)
    plt.close()

    # 图 3：平均误差 vs 初始偏角（全程 + 稳态）
    plt.figure(figsize=(8, 5))
    plt.plot(ANGLES_DEG, [avg_err[a] for a in ANGLES_DEG], marker='o', label="全程平均误差")
    plt.plot(ANGLES_DEG, [ss_err[a] for a in ANGLES_DEG], marker='s', label="稳态平均误差")
    plt.xlabel("初始偏角 (°)")
    plt.ylabel("平均误差 (°)")
    plt.title("三轴 RL 平均误差 vs 初始偏角")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "three_axis_mean_error.png"), dpi=150)
    plt.close()

    print(f"\n图片已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
