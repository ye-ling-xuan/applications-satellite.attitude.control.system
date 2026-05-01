#main函数：整个项目的入口，负责组装各个模块并执行一次完整的仿真实验。
from satellite import Satellite
from controller import PIDController
from simulator import run_simulation
from plotter import plot_results

# 创建卫星
sat = Satellite(I=1.0)
# 初始化卫星（初始角度，初始角速度）
sat.set_state(30.0)

# 创建PID控制器（尝试不同的kp,ki,kd值），dt一般为0.01s
pid = PIDController(Kp=3.0, Ki=0.5, Kd=1.0, dt=0.01)

# 运行仿真（传入卫星参数，pid参数，目标角度，调整允许的时间）
data = run_simulation(sat, pid, target_deg=0.0, duration=10.0)

# 绘制结果（传入刚才从仿真引擎得到的数据绘图）
#data 参数是一个字典，应包含键 'time'、'angle'、'omega'、'torque'
#每个键对应的值是一维 NumPy 数组或列表，长度相同。
plot_results(data)