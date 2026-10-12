#仿真引擎是整个卫星姿态控制系统的“总控中心”。
#它负责按时间顺序调用控制器和卫星动力学模型，记录数据，最终输出系统的响应曲线。
#没有仿真引擎，控制器和卫星模型只是孤立的部件，无法协同工作。

#仿真引擎的核心任务
#时间推进：从t=0 到 t=tend，以固定步长 Δt逐步前进。
#在每个时间步：
#1.获取卫星当前状态（角度、角速度）。
#2.调用控制器计算当前时刻需要的控制力矩。
#3.将该力矩传递给卫星动力学模型，更新卫星状态到下一时刻。
#4.记录所有感兴趣的数据（时间、状态、力矩、误差等）。
#5.输出：完整的时间序列数据，供可视化与性能分析。
import numpy as np
from satellite import Satellite
from controller import PIDController

#def 函数名称（参数列表）：函数体
#return返回结果

#sat：（satellite类的一个形参）卫星实例，包含当前状态（角度、角速度）和 apply_torque 方法。
#pid：（pidcontroller类的一个形参）PID控制器实例，包含 compute 方法和内部状态（积分、上一误差）。
#target_deg：期望角度（度），作为PID的设定值。
#duration：仿真总时长（秒）。
#dt：时间步长（秒），默认0.01秒。
def run_simulation(sat, pid, target_deg, duration, dt=0.01):
    steps = int(duration / dt)   #总步数
    #四个NumPy数组预先分配内存，避免在循环中动态追加，提高效率。后面用索引赋值。
    #这 4 行是 NumPy 里创建全零数组的核心语法
    #专门用来提前开辟内存、存储仿真过程中的数据（时间、角度、角速度、力矩）
    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)
    
    for i in range(steps):
        time[i] = i * dt  #记录当前时间：time[i] = i * dt
        angles[i] = sat.get_angle_deg()  #返回当前角度（度）
        omegas[i] = sat.get_omega_deg()  #返回当前角速度（度/秒）
        #pid 是之前传入的 PIDController 类的实例（对象）。
        #compute 是该类的一个方法，负责执行 PID 算法。
        #target_deg 是期望角度（设定值），例如 0°。
        #angles[i] 是当前时刻卫星的真实角度（测量值）。
        torque = pid.compute(target_deg, angles[i])  #torque翻译为力矩
        #torques[i] = torque 将其存入数组，用于后续分析
        torques[i] = torque
        #将这个力矩作用到卫星动力学模型上，更新卫星的角度和角速度。
        sat.apply_torque(torque, dt)
    
    #返回一个字典，方便后续绘图或性能分析。
    #可以用 results['time']、results['angle'] 等访问数据。
    return {
        'time': time,
        'angle': angles,
        'omega': omegas,
        'torque': torques
    }