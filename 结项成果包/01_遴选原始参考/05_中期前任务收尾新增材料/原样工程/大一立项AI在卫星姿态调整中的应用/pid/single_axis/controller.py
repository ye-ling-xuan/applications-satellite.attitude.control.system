import numpy as np
class PIDController:
    #保存 PID 增益、采样周期，并初始化积分项和上一时刻误差为 0。
    #注意：dt 必须与仿真步长一致（通常 0.01 s）。
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        #采样周期 (s)
        self.dt = dt
        #积分累加器（实际是 ∫e dt 的近似）
        self.integral = 0.0
        #上一次误差，用于微分计算
        self.prev_error = 0.0
    
    #输入为目标角度和当前角度，单位是 度
    def compute(self, target_deg, current_deg):
        #立即转换为 弧度
        target = np.radians(target_deg)
        current = np.radians(current_deg)
        #计算当前误差（弧度）
        error = target - current
        #积分累加：error * dt 是误差对时间的积分近似（矩形法）。
        #注意：这里 self.integral 实际存储的是 ∑ejΔt
        #即积分项的数值，不是纯积分值。
        self.integral += error * self.dt
        #微分项：用后向差分近似导数。
        #后向差分（误差2-误差1）/dt,代替连续函数的微分
        derivative = (error - self.prev_error) / self.dt
        #标准位置式 PID 输出，单位是 力矩（N·m）
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        #保存当前误差供下一步微分使用，并返回控制量
        self.prev_error = error
        return output
