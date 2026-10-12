import numpy as np

class SingleAxisSatellite:
    """单轴卫星动力学模型"""
    
    def __init__(self, I=1.0):
        """
        初始化卫星
        
        参数:
            I: 转动惯量 (kg·m²)
        """
        self.I = I
        self.theta = 0.0      # 角度 (rad)
        self.omega = 0.0      # 角速度 (rad/s)
    
    def set_state(self, theta_deg, omega_deg=0.0):
        """
        设置卫星状态
        
        参数:
            theta_deg: 初始角度 (度)
            omega_deg: 初始角速度 (度/秒)
        """
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)
    
    def update(self, torque, dt):
        """
        更新卫星状态（欧拉积分）
        
        参数:
            torque: 控制力矩 (N·m)
            dt: 时间步长 (s)
        """
        alpha = torque / self.I      # 角加速度
        self.omega += alpha * dt     # 更新角速度
        self.theta += self.omega * dt  # 更新角度
    
    def get_angle_deg(self):
        """获取当前角度（度）"""
        return np.degrees(self.theta)
    
    def get_omega_deg(self):
        """获取当前角速度（度/秒）"""
        return np.degrees(self.omega)
    
    def get_state_deg(self):
        """获取当前状态（度）"""
        return self.get_angle_deg(), self.get_omega_deg()