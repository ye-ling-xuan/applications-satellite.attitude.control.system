"""
统一性能评估函数（satellite_ai 新增模块）

ds10: 新增统一评估模块，PID 与强化学习共用同一套指标，定义对齐控制理论标准：
  - 稳定时间（进入 ±band 且之后持续稳定）
  - 超调量（绝对角度 deg 与百分比）
  - 稳态误差（末端平均绝对误差）
  - IAE / ITAE / ISE（误差积分指标）
  - 控制能耗（力矩平方积分 ∫τ²dt）
  - 成功率（末端是否收敛到目标带内）

原项目中该功能散落且定义不一致：
  - cms的代码批注/metrics.py（settling_time 用"未来50步窗口"硬编码）
  - yxy的学习笔记/接入gym的RL训练/compare_PID_PPO.py（settling_time 只取
    "首个进入容差时刻"，且 overshoot 对 target=0 会误算）
本模块统一口径，避免结论不可复现。
"""

import numpy as np


def compute_metrics(time, angle, omega=None, torque=None, target=0.0, band_deg=1.0):
    """
    计算一条轨迹的性能指标。

    参数:
        time  : 时间序列 (s)
        angle : 角度序列 (deg)
        omega : 角速度序列 (deg/s)，可选
        torque: 力矩序列 (N·m)，可选（用于能耗）
        target: 目标角度 (deg)
        band_deg: 稳定带半宽 (deg)，用于判定稳定时间与成功率

    返回:
        dict: 各项指标
    """
    time = np.asarray(time, dtype=float)
    angle = np.asarray(angle, dtype=float)
    e = angle - target
    n = len(e)
    dt = (time[1] - time[0]) if n > 1 else 0.01
    init_dev = abs(e[0]) if n > 0 else 0.0

    # ---- 稳定时间：进入 ±band 且之后持续稳定（不再超出） ----
    settle_idx = None
    for i in range(n):
        if abs(e[i]) <= band_deg and np.all(np.abs(e[i:]) <= band_deg):
            settle_idx = i
            break
    settling_time = float(time[settle_idx]) if settle_idx is not None else float('inf')

    # ---- 超调量：越过目标到另一侧的最大偏离 ----
    if init_dev > 1e-9:
        if e[0] > 0:
            overshoot_deg = max(0.0, float(-np.min(e)))
        else:
            overshoot_deg = max(0.0, float(np.max(e)))
        overshoot_pct = overshoot_deg / init_dev * 100.0
    else:
        overshoot_deg = 0.0
        overshoot_pct = 0.0

    # ---- 稳态误差：末端 10% 时间的平均绝对误差 ----
    tail = max(1, int(n * 0.1)) if n > 0 else 0
    steady_error = float(np.mean(np.abs(e[-tail:]))) if n > 0 else float('inf')

    # ---- 误差积分指标 ----
    iae = float(np.sum(np.abs(e)) * dt)
    itae = float(np.sum(time * np.abs(e)) * dt)
    ise = float(np.sum(e ** 2) * dt)

    # ---- 控制能耗 ----
    if torque is not None and len(torque) > 0:
        energy = float(np.sum(np.asarray(torque) ** 2) * dt)
    else:
        energy = float('nan')

    # ---- 成功率 ----
    success = bool(n > 0 and abs(e[-1]) <= band_deg)

    return {
        'settling_time': settling_time,
        'overshoot_deg': overshoot_deg,
        'overshoot_pct': overshoot_pct,
        'steady_error': steady_error,
        'IAE': iae,
        'ITAE': itae,
        'ISE': ise,
        'control_energy': energy,
        'success': success,
    }


def print_metrics(metrics, name=''):
    """打印单条轨迹的指标。"""
    print(f"\n--- {name} ---")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<16}: {val:.4f}")
        else:
            print(f"  {key:<16}: {val}")


def print_table(rows):
    """
    打印多组结果对比表。
    rows: [(label, metrics_dict), ...]
    """
    header = f"{'name':<16} {'settle(s)':>10} {'overshoot%':>12} "
    header += f"{'steady_err':>12} {'IAE':>10} {'energy':>10} {'success':>8}"
    print("\n" + header)
    print("-" * len(header))
    for label, m in rows:
        settle = 'inf' if not np.isfinite(m['settling_time']) else f"{m['settling_time']:.2f}"
        energy = 'nan' if np.isnan(m['control_energy']) else f"{m['control_energy']:.2f}"
        print(f"{label:<16} {settle:>10} {m['overshoot_pct']:>11.2f}% "
              f"{m['steady_error']:>12.4f} {m['IAE']:>10.2f} {energy:>10} {str(m['success']):>8}")
