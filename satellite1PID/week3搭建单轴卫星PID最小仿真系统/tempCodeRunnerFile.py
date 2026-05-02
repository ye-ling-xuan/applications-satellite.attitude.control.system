from satellite import Satellite
from controller import PIDController
from simulator import run_simulation
from plotter import plot_results

# 创建卫星
sat = Satellite(I=1.0)
sat.set_state(initial_angle_deg=30.0)

# 创建PID控制器
pid = PIDController(Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01)

# 运行仿真
data = run_simulation(sat, pid, target_deg=0.0, duration=10.0)

# 绘制结果
plot_results(data)