"""
三轴卫星 PID 控制仿真主程序。
演示从初始姿态（例如偏航30°）回归到零姿态。
"""
import numpy as np
from satellite3d import Satellite3D
from pid3d import PID3D
from simulator3d import run_simulation_3d
from plotter3d import plot_results_3d, plot_torque_3d
from quaternion_utils import euler_to_quaternion

def main():
    # 1. 创建卫星（默认转动惯量）
    sat = Satellite3D()

    # 设置初始姿态：偏航 30°，其它为0
    initial_q = euler_to_quaternion(0.0, 0.0, np.radians(30.0))
    sat.set_state(q=initial_q, omega=np.zeros(3))

    # 2. 创建 PID 控制器（三轴独立，增益相同）
    pid = PID3D(Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01,
                output_limit=2.0, integral_limit=5.0)

    # 3. 目标姿态：单位四元数（零姿态）
    target_q = np.array([1.0, 0.0, 0.0, 0.0])

    # 4. 定义干扰函数（正弦干扰，作用在三轴上）
    def disturbance_func(t):
        return 0.05 * np.array([np.sin(2*np.pi*0.2*t),
                                0.03 * np.sin(2*np.pi*0.3*t),
                                0.04 * np.sin(2*np.pi*0.25*t)])

    # 5. 运行仿真
    data = run_simulation_3d(
        sat, pid, target_q, duration=10.0, dt=0.01,
        disturbance_func=disturbance_func,
        noise_std_deg=0.3,       # 角度测量噪声 0.3°
        max_torque=2.0,          # 每个轴力矩饱和 ±2 N·m
        deadzone=0.02            # 死区 0.02 N·m
    )

    # 6. 绘图
    plot_results_3d(data)
    plot_torque_3d(data)

if __name__ == "__main__":
    main()