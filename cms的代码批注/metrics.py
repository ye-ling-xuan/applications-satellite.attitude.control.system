import numpy as np

def compute_metrics(angles, time):
    """
    计算PID控制器的性能指标
    
    参数:
        angles: 角度序列 (度)
        time: 时间序列 (秒)
    
    返回:
        dict: 包含各项性能指标
    """
    angles = np.array(angles)
    time = np.array(time)
    
    if len(angles) == 0:
        return {
            'settling_time': float('inf'),
            'overshoot': float('inf'),
            'steady_error': float('inf'),
            'control_energy': float('inf'),
            'peak_time': float('inf'),
            'rise_time': float('inf')
        }
    
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    target_angle = 0.0
    
    # 1. 稳定时间（进入±1度且不再超出）
    settle_tolerance = 1.0
    settle_start = None
    settled = False
    
    for i in range(len(angles)):
        if abs(angles[i] - target_angle) < settle_tolerance:
            if not settled:
                # 检查后续是否持续稳定
                future_checks = min(50, len(angles) - i)
                if all(abs(angles[i:i+future_checks] - target_angle) < settle_tolerance):
                    settle_start = time[i]
                    settled = True
        else:
            settled = False
    
    settling_time = settle_start if settle_start is not None else time[-1]
    
    # 2. 超调量（最大偏离）
    if target_angle == 0:
        overshoot = np.max(np.abs(angles))
    else:
        overshoot = np.max(np.abs(angles - target_angle))
    
    # 3. 稳态误差（最后1秒的平均值）
    last_second = int(1.0 / dt)
    if last_second < len(angles):
        steady_error = np.mean(np.abs(angles[-last_second:] - target_angle))
    else:
        steady_error = np.mean(np.abs(angles - target_angle))
    
    # 4. 上升时间（从10%到90%）
    initial_angle = abs(angles[0] - target_angle)
    target_10 = 0.1 * initial_angle
    target_90 = 0.9 * initial_angle
    
    rise_start_time = None
    rise_end_time = None
    
    for i in range(len(angles)):
        current_deviation = abs(angles[i] - target_angle)
        if rise_start_time is None and current_deviation <= target_10:
            rise_start_time = time[i]
        if rise_end_time is None and current_deviation <= target_90:
            rise_end_time = time[i]
    
    rise_time = (rise_end_time - rise_start_time) if (rise_start_time and rise_end_time) else time[-1]
    
    # 5. 峰值时间（第一次达到峰值的时间）
    peak_idx = np.argmax(np.abs(angles))
    peak_time = time[peak_idx] if peak_idx < len(time) else time[-1]
    
    # 6. 控制能量（力矩的平方积分，需要从外部传入）
    # 这个值需要单独计算，因为需要力矩数据
    
    return {
        'settling_time': settling_time,
        'overshoot': overshoot,
        'steady_error': steady_error,
        'rise_time': rise_time,
        'peak_time': peak_time
    }


def compute_full_metrics(angles, time, torques=None):
    """
    计算完整的性能指标（包含能耗）
    
    参数:
        angles: 角度序列 (度)
        time: 时间序列 (秒)
        torques: 力矩序列 (N·m)，可选
    
    返回:
        dict: 包含所有性能指标
    """
    metrics = compute_metrics(angles, time)
    
    # 计算控制能量
    if torques is not None and len(torques) > 0:
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        control_energy = np.sum(np.array(torques)**2) * dt
        metrics['control_energy'] = control_energy
    else:
        metrics['control_energy'] = np.nan
    
    return metrics


def print_metrics_table(metrics_dict, param_label='Parameters'):
    """打印性能指标表格"""
    print("\n" + "="*80)
    print(f"{param_label:^20} | 稳定时间(s) | 超调(°) | 稳态误差(°) | 上升时间(s) | 峰值时间(s) | 能耗")
    print("-"*80)
    
    for label, m in metrics_dict.items():
        print(f"{label:^20} | {m['settling_time']:^11.2f} | {m['overshoot']:^8.2f} | {m['steady_error']:^12.4f} | {m['rise_time']:^11.2f} | {m['peak_time']:^11.2f} | {m['control_energy']:.2f}")
    
    print("="*80)


def meets_target(metrics, settling_target=3.0, overshoot_target=10.0):
    """
    检查是否满足整定目标
    
    参数:
        metrics: 性能指标字典
        settling_target: 稳定时间目标(秒)
        overshoot_target: 超调量目标(%)
    
    返回:
        bool: 是否满足目标
    """
    return (metrics['settling_time'] <= settling_target and 
            metrics['overshoot'] <= overshoot_target)