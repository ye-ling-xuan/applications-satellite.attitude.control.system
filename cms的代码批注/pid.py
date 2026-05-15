import numpy as np

class PIDController:
    """离散PID控制器"""
    
    def __init__(self, Kp, Ki, Kd, dt, output_limit=None, integral_limit=None):
        """
        初始化PID控制器
        
        参数:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
            dt: 采样时间 (s)
            output_limit: 输出限幅 (max_torque)
            integral_limit: 积分限幅
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
        self.integral = 0.0      # 积分项
        self.prev_error = 0.0    # 上一次误差
    
    def compute(self, setpoint, measurement):
        """
        计算PID输出（位置式）
        
        参数:
            setpoint: 目标值 (度)
            measurement: 测量值 (度)
        
        返回:
            output: 控制输出 (N·m)
        """
        # 计算误差（处理角度绕圈问题）
        error = self._angle_error(setpoint, measurement)
        
        # 比例项
        P = self.Kp * error
        
        # 积分项（带限幅）
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self.integral
        
        # 微分项
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative
        
        # 总输出
        output = P + I + D
        
        # 输出限幅
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        
        # 保存误差供下次使用
        self.prev_error = error
        
        return output
    
    def _angle_error(self, target, current):
        """
        计算角度误差（处理-180°到180°的边界）
        
        参数:
            target: 目标角度 (度)
            current: 当前角度 (度)
        
        返回:
            error: 最短弧长误差 (度)
        """
        error = target - current
        # 规整到[-180, 180]
        error = (error + 180) % 360 - 180
        return error
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0