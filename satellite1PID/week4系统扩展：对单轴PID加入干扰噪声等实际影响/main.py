import numpy as np                      # 新增：用于定义干扰函数
from satellite import Satellite
from controller import PIDController
from simulator import run_simulation
from plotter import plot_results

# 1. 创建卫星
sat = Satellite(I=1.0)
sat.set_state(30.0)                     # 初始角度30度，角速度默认0

# 2. 创建PID控制器（带积分限幅和输出限幅）
pid = PIDController(
    Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01,
    output_limit=2.0,                   # 限制最大输出力矩 ±2 N·m
    integral_limit=5.0                  # 限制积分累加器在 ±5 rad·s
)

# 3. 定义干扰函数（正弦波，幅值0.05 N·m，频率0.2 Hz）
def disturbance_func(t):
    return 0.05 * np.sin(2 * np.pi * 0.2 * t)

# 4. 运行仿真（使用所有扩展功能）
data = run_simulation(
    sat, pid, target_deg=0.0, duration=10.0, dt=0.01,
    disturbance_func=disturbance_func,   # 加入干扰
    noise_std_deg=0.3,                   # 角度测量噪声标准差0.3度
    max_torque=2.0,                      # 执行器力矩饱和限制
    deadzone=0.02                        # 执行器死区0.02 N·m
)

# 5. 绘制结果（绘图函数已适配新数据格式）
plot_results(data)