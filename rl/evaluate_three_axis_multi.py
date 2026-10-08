"""
三轴 RL 多轴评测：
  A. 单轴参考（绕 roll / pitch / yaw 各自 0–90°）—— 验证三轴环境对每个轴都有效
  B. 三轴同时偏转（对称 (θ,θ,θ) 与非对称耦合姿态）—— 真正的三轴机动
测总指向误差的纠正时间 / 平均误差，出图并导出 CSV。
用法: cd rl && python evaluate_three_axis_multi.py
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

# 测试姿态（单位：度，ZYX 欧拉角 roll/pitch/yaw）
AXIS_SWEEPS = {
    "roll":  [(a, 0, 0) for a in range(0, 91, 15)],
    "pitch": [(0, a, 0) for a in range(0, 91, 15)],
    "yaw":   [(0, 0, a) for a in range(0, 91, 15)],
}
SYMMETRIC = [(t, t, t) for t in range(0, 61, 15)]          # 0,15,30,45,60
ASYMMETRIC = [(45, -30, 45), (50, 30, -40), (30, 60, 20)]

THRESHOLD_DEG = 1.0
MAX_STEPS = 2000


def run_episode(model, env, euler_deg):
    """从指定三轴欧拉角（度）跑一条确定性轨迹，返回 (times, total_error_deg)。"""
    q0 = _euler_to_quat(*np.radians(euler_deg))
    obs, _ = env.reset(options={'q': q0, 'omega': np.zeros(3)})
    times, errors = [], []
    dt = env.dt
    done = False
    step = 0
    times.append(0.0)
    errors.append(float(np.degrees(np.linalg.norm(obs[:3]))))
    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        times.append(step * dt)
        errors.append(float(np.degrees(np.linalg.norm(obs[:3]))))
    return np.array(times), np.array(errors)


def main():
    model = PPO.load(MODEL_PATH)
    env = SatelliteEnv3D(max_steps=MAX_STEPS)

    # 组装所有测试用例：(类别, 欧拉角)
    cases = []
    for cat, sweep in AXIS_SWEEPS.items():
        for e in sweep:
            cases.append((cat, e))
    for e in SYMMETRIC:
        cases.append(("三轴对称", e))
    for e in ASYMMETRIC:
        cases.append(("三轴耦合", e))

    rows = []  # (cat, euler, init_err, settle, mean_err, ss_err, times, errors)
    for cat, e in cases:
        t, err = run_episode(model, env, e)
        st = settling_time(err, t, THRESHOLD_DEG)
        ss = steady_state_error(err, t, THRESHOLD_DEG)
        rows.append((cat, e, err[0], st, mean_error(err), ss, t, err))

    print(f"{'类型':<10} {'姿态(roll,pitch,yaw)':>22} | {'初始总误差':>10} | {'纠正时间':>10} | {'稳态均误':>10}")
    for cat, e, init_err, st, me, ss, _, _ in rows:
        s = f"{st:.2f}s" if st is not None else "未稳定"
        s2 = f"{ss:.2f}°" if not np.isnan(ss) else "——"
        print(f"{cat:<10} {str(e):>22} | {init_err:>9.1f}° | {s:>10} | {s2:>10}")

    # 导出结果表
    csv_path = os.path.join(OUT_DIR, "three_axis_multi_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类型", "roll(°)", "pitch(°)", "yaw(°)",
                    "初始总指向误差(°)", "纠正时间(s)", "全程平均误差(°)", "稳态平均误差(°)"])
        for cat, e, init_err, st, me, ss, _, _ in rows:
            w.writerow([cat, e[0], e[1], e[2],
                        f"{init_err:.4f}",
                        f"{st:.4f}" if st is not None else "",
                        f"{me:.4f}",
                        f"{ss:.4f}" if not np.isnan(ss) else ""])

    # 图 1：代表姿态的纠正曲线（总指向误差 vs 时间）
    plt.figure(figsize=(10, 5))
    for cat, e, _, _, _, _, t, err in rows:
        if e in [(0, 0, 60), (45, 45, 45), (50, 30, -40)]:
            plt.plot(t, err, label=f"{cat} {e}°")
    plt.axhline(THRESHOLD_DEG, color='gray', linestyle='--', label=f"阈值 {THRESHOLD_DEG}°")
    plt.xlabel("时间 (s)")
    plt.ylabel("总指向误差 (°)")
    plt.title("三轴 RL 纠正曲线（含单轴参考与多轴耦合）")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "three_axis_multi_correction_curves.png"), dpi=150)
    plt.close()

    # 图 2：纠正时间 vs 初始总指向误差
    plt.figure(figsize=(8, 5))
    markers = {"roll": "v", "pitch": "^", "yaw": "o", "三轴对称": "s", "三轴耦合": "*"}
    for cat in markers:
        xs = [r[2] for r in rows if r[0] == cat]
        ys = [r[3] if r[3] is not None else float('nan') for r in rows if r[0] == cat]
        plt.plot(xs, ys, marker=markers[cat], linestyle='', label=cat)
    plt.xlabel("初始总指向误差 (°)")
    plt.ylabel("纠正时间 (s)")
    plt.title("三轴 RL 纠正时间 vs 初始总指向误差")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "three_axis_multi_settling_time.png"), dpi=150)
    plt.close()

    # 图 3：稳态平均误差 vs 初始总指向误差
    plt.figure(figsize=(8, 5))
    for cat in markers:
        xs = [r[2] for r in rows if r[0] == cat]
        ys = [r[5] if not np.isnan(r[5]) else float('nan') for r in rows if r[0] == cat]
        plt.plot(xs, ys, marker=markers[cat], linestyle='', label=cat)
    plt.xlabel("初始总指向误差 (°)")
    plt.ylabel("稳态平均误差 (°)")
    plt.title("三轴 RL 稳态平均误差 vs 初始总指向误差")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "three_axis_multi_steady_error.png"), dpi=150)
    plt.close()

    print(f"\n图片已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
