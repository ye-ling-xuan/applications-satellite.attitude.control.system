"""
RL 算法对比：PPO vs SAC vs TD3（三轴姿态控制，无干扰干净对比）。

统一用 三轴强化学习/satellite_env3d.py 的动力学作为唯一仿真器，对同一组 29 个初始姿态
分别跑 PPO / SAC / TD3 三个训练好的策略，比较纠正时间与稳态误差。
三个模型均用相同的环境、奖励函数、随机初始化与训练预算（80 万步）。

用法: cd RL与PID对比 && python compare_rl_algorithms.py
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
from stable_baselines3 import PPO, SAC, TD3

HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.join(HERE, "..", "三轴强化学习")
sys.path.insert(0, RL_DIR)

from satellite_env3d import SatelliteEnv3D, _euler_to_quat  # noqa: E402  # type: ignore[import-not-found]
from metrics import settling_time  # noqa: E402  # type: ignore[import-not-found]

OUT_DIR = os.path.join(HERE, "成果图")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD_DEG = 1.0
MAX_STEPS = 2000
DT = 0.01

# 测试姿态（度，ZYX 欧拉角 roll/pitch/yaw），与 RL vs PID 对比完全一致
AXIS_SWEEPS = {
    "roll":  [(a, 0, 0) for a in range(0, 91, 15)],
    "pitch": [(0, a, 0) for a in range(0, 91, 15)],
    "yaw":   [(0, 0, a) for a in range(0, 91, 15)],
}
SYMMETRIC = [(t, t, t) for t in range(0, 61, 15)]          # 0,15,30,45,60
ASYMMETRIC = [(45, -30, 45), (50, 30, -40), (30, 60, 20)]

ALGOS = {
    "PPO": os.path.join(RL_DIR, "ppo_satellite3d"),
    "SAC": os.path.join(RL_DIR, "sac_satellite3d"),
    "TD3": os.path.join(RL_DIR, "td3_satellite3d"),
}
LOADERS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
COLORS = {"PPO": "#4C72B0", "SAC": "#55A868", "TD3": "#C44E52"}
MARKERS = {"PPO": "o", "SAC": "s", "TD3": "^"}


def run_episode(env, model, euler_deg):
    """从指定三轴欧拉角（度）跑一条轨迹，返回 (times, total_error_deg)。"""
    q0 = _euler_to_quat(*np.radians(euler_deg))
    obs, _ = env.reset(options={'q': q0, 'omega': np.zeros(3)})
    times, errors = [0.0], [float(np.degrees(np.linalg.norm(obs[:3])))]
    done = False
    step = 0
    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        times.append(step * DT)
        errors.append(float(np.degrees(np.linalg.norm(obs[:3]))))
    return np.array(times), np.array(errors)


TAIL_STEPS = 50  # 末段窗口：0.5 s


def tail_error(errors):
    """末段平均误差（度）：轨迹最后 0.5 s 的 |误差| 均值。

    不用 metrics.steady_state_error：它取"最后一次超过阈值之后"的均值，
    对**持续振荡、始终不收敛**的策略会退化成只剩 1 个采样点的平均（数值虚低）。
    末段均值才是不收敛策略也适用的、可比的稳态精度指标。
    """
    return float(np.mean(np.asarray(errors)[-TAIL_STEPS:]))


def evaluate(model):
    """对全部 29 个测试姿态跑一遍，返回 rows（每项: cat, euler, init_err, settle, tail_err, times, errors）。"""
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
        t, err = run_episode(env, model, e)
        st = settling_time(err, t, THRESHOLD_DEG)
        ss = tail_error(err)
        rows.append((cat, e, err[0], st, ss, t, err))
    return rows


def main():
    # 依次加载并评测三个模型
    rows = {}
    for name in ALGOS:
        print(f"评测 {name} ...")
        model = LOADERS[name].load(ALGOS[name])
        rows[name] = evaluate(model)

    # 导出 CSV
    csv_path = os.path.join(OUT_DIR, "rl_algorithms_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["类型", "roll(°)", "pitch(°)", "yaw(°)", "初始总误差(°)"]
        for name in ALGOS:
            header.append(f"{name}纠正时间(s)")
        for name in ALGOS:
            header.append(f"{name}末段平均误差(°)")
        w.writerow(header)
        for i, (cat, e, init, _, _, _, _) in enumerate(rows["PPO"]):
            line = [cat, e[0], e[1], e[2], f"{init:.4f}"]
            for name in ALGOS:
                st = rows[name][i][3]
                line.append(f"{st:.4f}" if st is not None else "")
            for name in ALGOS:
                line.append(f"{rows[name][i][4]:.4f}")
            w.writerow(line)

    # 汇总表
    # 注：纠正时间只统计"初始误差 > 阈值"的姿态——初始就在阈值内的姿态（如 0°）本身
    #     就算"稳定"，把它们平均进去会把均值拉成 0.00 s，产生误导。
    print("\n汇总（29 个姿态）:")
    print(f"{'算法':<6} {'纠正时间均值':>12} {'末段平均误差':>12} {'未稳定':>10}")
    for name in ALGOS:
        nontrivial = [r for r in rows[name] if r[2] > THRESHOLD_DEG]
        settles = [r[3] for r in nontrivial if r[3] is not None]
        tails = [r[4] for r in rows[name]]
        n_fail = sum(1 for r in nontrivial if r[3] is None)
        avg_st = f"{np.mean(settles):.2f}s" if settles else "—"
        print(f"{name:<6} {avg_st:>12} {np.mean(tails):>11.3f}° {n_fail:>6}/{len(nontrivial)}")

    # 图 1：代表姿态纠正曲线对比
    targets = [(0, 0, 60), (45, 45, 45), (50, 30, -40)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, target in zip(axes, targets):
        for name in ALGOS:
            for cat, e, _, _, _, t, err in rows[name]:
                if e == target:
                    ax.plot(t, err, color=COLORS[name], label=name)
        ax.axhline(THRESHOLD_DEG, color='gray', linestyle='--', label=f"阈值 {THRESHOLD_DEG}°")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("总指向误差 (°)")
        ax.set_title(f"姿态 {target}°")
        ax.grid(True)
        ax.legend()
    fig.suptitle("PPO vs SAC vs TD3 纠正曲线")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rl_algorithms_curves.png"), dpi=150)
    plt.close(fig)

    # 图 2：纠正时间散点图（无折线）
    cap = MAX_STEPS * DT  # 20 s，超过即视为未稳定
    plt.figure(figsize=(8, 5))
    for name in ALGOS:
        x_ok = [r[2] for r in rows[name] if r[3] is not None]
        y_ok = [r[3] for r in rows[name] if r[3] is not None]
        x_fail = [r[2] for r in rows[name] if r[3] is None]
        plt.scatter(x_ok, y_ok, s=60, marker=MARKERS[name], color=COLORS[name], label=name)
        if x_fail:
            plt.scatter(x_fail, [cap] * len(x_fail), s=90, marker='x', color=COLORS[name],
                        label=f'{name} 未稳定({len(x_fail)})')
            plt.axhline(cap, color=COLORS[name], linestyle=':', alpha=0.4)
    plt.xlabel("初始总指向误差 (°)")
    plt.ylabel("纠正时间 (s)")
    plt.title("PPO vs SAC vs TD3 纠正时间散点图")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rl_algorithms_settling_time.png"), dpi=150)
    plt.close()

    # 图 3：末段平均误差散点图（无折线）
    plt.figure(figsize=(8, 5))
    for name in ALGOS:
        x = [r[2] for r in rows[name]]
        y = [r[4] for r in rows[name]]
        plt.scatter(x, y, s=60, marker=MARKERS[name], color=COLORS[name], label=name)
    plt.axhline(THRESHOLD_DEG, color='gray', linestyle='--', label=f"阈值 {THRESHOLD_DEG}°")
    plt.xlabel("初始总指向误差 (°)")
    plt.ylabel("末段平均误差 (°, 末 0.5 s)")
    plt.title("PPO vs SAC vs TD3 末段平均误差散点图")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rl_algorithms_tail_error.png"), dpi=150)
    plt.close()

    print(f"\n图片与 CSV 已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
