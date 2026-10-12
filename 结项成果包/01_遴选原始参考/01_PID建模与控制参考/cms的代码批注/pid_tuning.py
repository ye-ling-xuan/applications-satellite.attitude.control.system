import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import product
from satellite import SingleAxisSatellite
from pid import PIDController

# 导入性能指标函数
from metrics import compute_full_metrics, print_metrics_table, meets_target


def run_single_simulation(Kp, Ki, Kd, initial_angle_deg=30.0, duration=10.0, dt=0.01):
    """
    运行单次PID仿真
    
    返回:
        dict: 包含时间、角度、力矩的历史数据
    """
    # 创建卫星和控制器
    sat = SingleAxisSatellite(I=1.0)
    sat.set_state(theta_deg=initial_angle_deg)
    
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, output_limit=2.0)
    
    # 预分配数组
    steps = int(duration / dt)
    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)
    
    # 仿真循环
    for i in range(steps):
        time[i] = i * dt
        angles[i] = sat.get_angle_deg()
        omegas[i] = sat.get_omega_deg()
        
        torque = pid.compute(0.0, angles[i])
        torques[i] = torque
        
        sat.update(torque, dt)
    
    return {
        'time': time,
        'angle': angles,
        'omega': omegas,
        'torque': torques
    }


def batch_tuning():
    """
    批量运行参数组合，自动记录性能指标
    """
    print("="*60)
    print("PID参数整定开始")
    print("="*60)
    
    # 参数范围
    Kp_range = [1.0, 2.0, 3.0, 4.0, 5.0]
    Ki_range = [0.0, 0.2, 0.5, 0.8, 1.0]
    Kd_range = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print(f"\n参数范围:")
    print(f"  Kp: {Kp_range}")
    print(f"  Ki: {Ki_range}")
    print(f"  Kd: {Kd_range}")
    print(f"  总组合数: {len(Kp_range) * len(Ki_range) * len(Kd_range)}")
    
    # 存储结果
    results = []
    
    # 批量运行
    for Kp, Ki, Kd in product(Kp_range, Ki_range, Kd_range):
        # 运行仿真
        data = run_single_simulation(Kp, Ki, Kd)
        
        # 计算性能指标
        metrics = compute_full_metrics(
            angles=data['angle'],
            time=data['time'],
            torques=data['torque']
        )
        
        # 记录结果
        result = {
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'metrics': metrics
        }
        results.append(result)
        
        # 实时打印进度
        if len(results) % 10 == 0:
            print(f"已完成 {len(results)}/{len(Kp_range)*len(Ki_range)*len(Kd_range)} 组参数")
    
    print("\n批量仿真完成！")
    
    return results, Kp_range, Ki_range, Kd_range


def find_best_parameters(results, settling_target=3.0, overshoot_target=10.0):
    """
    根据性能指标选出最优参数
    """
    # 筛选满足目标的参数
    qualified = []
    for r in results:
        m = r['metrics']
        if (m['settling_time'] <= settling_target and 
            m['overshoot'] <= overshoot_target):
            qualified.append(r)
    
    if not qualified:
        print(f"\n警告：没有找到同时满足稳定时间<{settling_target}s和超调<{overshoot_target}%的参数")
        print("将使用综合评分最低的参数")
        
        # 计算综合评分（归一化后加权和）
        for r in results:
            m = r['metrics']
            # 综合评分：稳定时间 + 超调/10 + 稳态误差*10 + 能耗/10
            score = (m['settling_time'] + 
                    m['overshoot']/10 + 
                    m['steady_error']*10 + 
                    m['control_energy']/10)
            r['score'] = score
        
        best = min(results, key=lambda x: x['score'])
        return best
    
    # 在满足条件的参数中，选择能耗最低的
    best = min(qualified, key=lambda x: x['metrics']['control_energy'])
    
    return best


def plot_parameter_analysis(results, Kp_range, Ki_range, Kd_range):
    """
    绘制参数对性能指标的影响图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 稳定时间 vs Kp (固定 Ki=0.5, Kd=1.0)
    fixed_Ki, fixed_Kd = 0.5, 1.0
    kp_vals = []
    settling_times = []
    overshoots = []
    energies = []
    
    for Kp in Kp_range:
        for r in results:
            if (r['Kp'] == Kp and r['Ki'] == fixed_Ki and r['Kd'] == fixed_Kd):
                kp_vals.append(Kp)
                settling_times.append(r['metrics']['settling_time'])
                overshoots.append(r['metrics']['overshoot'])
                energies.append(r['metrics']['control_energy'])
                break
    
    axes[0, 0].plot(kp_vals, settling_times, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=3.0, color='r', linestyle='--', label='目标(3s)')
    axes[0, 0].set_xlabel('Kp')
    axes[0, 0].set_ylabel('稳定时间 (s)')
    axes[0, 0].set_title('稳定时间 vs 比例增益 Kp')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 超调 vs Kd (固定 Kp=3.0, Ki=0.5)
    fixed_Kp, fixed_Ki = 3.0, 0.5
    kd_vals = []
    overshoots_kd = []
    
    for Kd in Kd_range:
        for r in results:
            if (r['Kp'] == fixed_Kp and r['Ki'] == fixed_Ki and r['Kd'] == Kd):
                kd_vals.append(Kd)
                overshoots_kd.append(r['metrics']['overshoot'])
                break
    
    axes[0, 1].plot(kd_vals, overshoots_kd, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=10.0, color='g', linestyle='--', label='目标(10%)')
    axes[0, 1].set_xlabel('Kd')
    axes[0, 1].set_ylabel('超调 (%)')
    axes[0, 1].set_title('超调 vs 微分增益 Kd')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 稳态误差 vs Ki (固定 Kp=3.0, Kd=1.0)
    fixed_Kp, fixed_Kd = 3.0, 1.0
    ki_vals = []
    steady_errors = []
    
    for Ki in Ki_range:
        for r in results:
            if (r['Kp'] == fixed_Kp and r['Ki'] == Ki and r['Kd'] == fixed_Kd):
                ki_vals.append(Ki)
                steady_errors.append(r['metrics']['steady_error'])
                break
    
    axes[0, 2].plot(ki_vals, steady_errors, 'go-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Ki')
    axes[0, 2].set_ylabel('稳态误差 (°)')
    axes[0, 2].set_title('稳态误差 vs 积分增益 Ki')
    axes[0, 2].grid(True)
    
    # 4. 控制能耗 vs Kp
    axes[1, 0].plot(kp_vals, energies, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Kp')
    axes[1, 0].set_ylabel('控制能耗')
    axes[1, 0].set_title('控制能耗 vs 比例增益 Kp')
    axes[1, 0].grid(True)
    
    # 5. 稳定时间 vs Kd
    axes[1, 1].plot(kd_vals, [r['metrics']['settling_time'] for r in results if r['Kp']==fixed_Kp and r['Ki']==fixed_Ki], 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Kd')
    axes[1, 1].set_ylabel('稳定时间 (s)')
    axes[1, 1].set_title('稳定时间 vs 微分增益 Kd')
    axes[1, 1].grid(True)
    
    # 6. 综合性能热力图 (Kp vs Ki, 固定 Kd=1.0)
    fixed_Kd = 1.0
    heatmap_data = np.zeros((len(Kp_range), len(Ki_range)))
    for i, Kp in enumerate(Kp_range):
        for j, Ki in enumerate(Ki_range):
            for r in results:
                if r['Kp'] == Kp and r['Ki'] == Ki and r['Kd'] == fixed_Kd:
                    # 综合评分
                    m = r['metrics']
                    score = m['settling_time'] + m['overshoot']/10 + m['steady_error']*10
                    heatmap_data[i, j] = score
                    break
    
    im = axes[1, 2].imshow(heatmap_data, origin='lower', 
                            extent=[Ki_range[0], Ki_range[-1], Kp_range[0], Kp_range[-1]],
                            aspect='auto', cmap='RdYlGn_r')
    axes[1, 2].set_xlabel('Ki')
    axes[1, 2].set_ylabel('Kp')
    axes[1, 2].set_title(f'综合性能热力图 (Kd={fixed_Kd})')
    plt.colorbar(im, ax=axes[1, 2], label='综合评分(越低越好)')
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_best_config(best_params, filename='config.json'):
    """
    将最优参数保存到配置文件
    """
    config = {
        'pid_parameters': {
            'Kp': best_params['Kp'],
            'Ki': best_params['Ki'],
            'Kd': best_params['Kd']
        },
        'performance': best_params['metrics'],
        'simulation_settings': {
            'dt': 0.01,
            'max_torque': 2.0,
            'target_angle': 0.0
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"\n最优参数已保存到 {filename}")


def plot_best_response(best_params):
    """
    绘制最优参数下的响应曲线
    """
    # 运行仿真
    data = run_single_simulation(
        Kp=best_params['Kp'],
        Ki=best_params['Ki'],
        Kd=best_params['Kd']
    )
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # 角度响应
    axes[0].plot(data['time'], data['angle'], 'b-', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='目标(0°)')
    axes[0].axhline(y=1, color='g', linestyle=':', alpha=0.5, label='±1°容差')
    axes[0].axhline(y=-1, color='g', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('角度 (度)')
    axes[0].set_title(f'最优PID响应 (Kp={best_params["Kp"]}, Ki={best_params["Ki"]}, Kd={best_params["Kd"]})')
    axes[0].legend()
    axes[0].grid(True)
    
    # 角速度
    axes[1].plot(data['time'], data['omega'], 'g-', linewidth=2)
    axes[1].set_ylabel('角速度 (度/秒)')
    axes[1].grid(True)
    
    # 控制力矩
    axes[2].plot(data['time'], data['torque'], 'r-', linewidth=2)
    axes[2].set_xlabel('时间 (秒)')
    axes[2].set_ylabel('控制力矩 (N·m)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('best_pid_response.png', dpi=150)
    plt.show()
    
    # 打印性能指标
    print("\n" + "="*50)
    print("最优PID参数性能指标")
    print("="*50)
    metrics = best_params['metrics']
    print(f"稳定时间:  {metrics['settling_time']:.2f} 秒")
    print(f"超调:      {metrics['overshoot']:.2f} 度")
    print(f"稳态误差:  {metrics['steady_error']:.4f} 度")
    print(f"上升时间:  {metrics['rise_time']:.2f} 秒")
    print(f"峰值时间:  {metrics['peak_time']:.2f} 秒")
    print(f"控制能耗:  {metrics['control_energy']:.2f}")


# ========== 主程序 ==========
if __name__ == "__main__":
    # 1. 批量运行参数组合
    results, Kp_range, Ki_range, Kd_range = batch_tuning()
    
    # 2. 找出最优参数
    best = find_best_parameters(results, settling_target=3.0, overshoot_target=10.0)
    
    print("\n" + "="*60)
    print("最优参数:")
    print(f"  Kp = {best['Kp']}")
    print(f"  Ki = {best['Ki']}")
    print(f"  Kd = {best['Kd']}")
    print("="*60)
    
    # 3. 绘制参数影响分析图
    plot_parameter_analysis(results, Kp_range, Ki_range, Kd_range)
    
    # 4. 绘制最优参数响应曲线
    plot_best_response(best)
    
    # 5. 保存配置
    save_best_config(best, 'config.json')