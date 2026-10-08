"""
评测指标：纠正时间（settling time）与平均误差。
"""
import numpy as np


def settling_time(errors_deg, times, threshold_deg=1.0):
    """
    纠正时间：误差首次进入并此后一直保持在阈值内的时刻（秒）。

    参数:
        errors_deg: 逐步误差幅值（度），一维数组
        times: 逐步时间（秒），一维数组（与 errors_deg 等长）
        threshold_deg: 判定"已纠正"的阈值（度）

    返回:
        纠正时间（秒）；若全程都未稳定到阈值内，返回 None。
    """
    errors = np.asarray(errors_deg, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    exceed = np.where(errors > threshold_deg)[0]
    if exceed.size == 0:
        # 全程都在阈值内 → 从第 0 秒就算稳定
        return float(times[0])
    settle_idx = exceed[-1] + 1
    if settle_idx >= len(times):
        return None  # 从未稳定
    return float(times[settle_idx])


def mean_error(errors_deg, start_frac=0.0, end_frac=1.0):
    """
    平均误差：给定时间段内 |误差|（度）的平均。

    参数:
        errors_deg: 逐步误差幅值（度）
        start_frac / end_frac: 取 [start_frac, end_frac) 这段比例的时间窗
    """
    errors = np.asarray(errors_deg, dtype=np.float64)
    n = len(errors)
    a = int(start_frac * n)
    b = int(end_frac * n)
    if b <= a:
        b = n
    return float(np.mean(np.abs(errors[a:b])))


def steady_state_error(errors_deg, times, threshold_deg=1.0):
    """
    稳态平均误差：纠正稳定后（误差此后一直 ≤ 阈值）的 |误差| 平均（度）。

    参数:
        errors_deg: 逐步误差幅值（度）
        times: 逐步时间（秒）
        threshold_deg: 判定"已稳定"的阈值（度）

    返回:
        稳定后的平均误差（度）；若全程都未稳定，返回 nan。
    """
    errors = np.asarray(errors_deg, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    exceed = np.where(errors > threshold_deg)[0]
    if exceed.size == 0:
        return float(np.mean(np.abs(errors)))
    settle_idx = exceed[-1] + 1
    if settle_idx >= len(errors):
        return float('nan')
    return float(np.mean(np.abs(errors[settle_idx:])))
