import numpy as np
class PIDController:
    """
    离散位置式 PID 控制器，支持积分限幅和输出限幅。
    
    参数：
        Kp, Ki, Kd: PID 增益
        dt: 采样周期 (s)
        output_limit: 输出力矩绝对值上限 (N·m)，默认 None（不限幅）
        integral_limit: 积分累加器绝对值上限 (rad·s)，默认 None（不限幅）
    """
    def __init__(self, Kp, Ki, Kd, dt, 
                 output_limit=None, 
                 integral_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        #采样周期 (s)
        self.dt = dt
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0        # 积分累加器 (rad·s)
        self.prev_error = 0.0      # 上一次误差 (rad)
    
    def compute(self, target_deg, current_deg):
        """
        计算控制力矩。
        输入：目标角度 (度)，当前测量角度 (度)
        输出：控制力矩 (N·m)
        """
        #立即转换为 弧度
        target = np.radians(target_deg)
        current = np.radians(current_deg)
        #计算当前误差（弧度）
        error = target - current

        '''
        积分累加：error * dt 是误差对时间的积分近似（矩形法）。
        注意：这里 self.integral 实际存储的是 ∑ejΔt
        即积分项的数值，不是纯积分值。
        '''
        self.integral += error * self.dt

        '''
        微分项：用后向差分近似导数。
        后向差分（误差2-误差1）/dt,代替连续函数的微分
        '''
        derivative = (error - self.prev_error) / self.dt
        
        #标准位置式 PID 输出，单位是 力矩（N·m）
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        
        #保存当前误差供下一步微分使用，并返回控制量
        self.prev_error = error
        return output
def reset(self):
        """重置控制器内部状态（积分项和上一误差），用于多次独立仿真"""
        self.integral = 0.0
        self.prev_error = 0.0
