#卫星动力学类设计与实现
#目标：构建一个可靠的单轴卫星姿态动力学模型，能够根据外力矩更新角度和角速度。
#核心数学：欧拉动力学（转动定律）与数值积分（欧拉法）。
#产出：Satellite1D 类，支持设置初始状态、欧拉积分更新、获取当前状态。

#引用numpy库
import numpy as np
#定义一个名为Satellite的类
class Satellite:
    #I=1.0创建对象时，如果不手动传入 I 的值，就自动用 1.0 作为初始值
    def __init__(self, I=1.0):
        self.I = I
        # theta和Omega默认不可配置：创建对象时不能直接修改，只能后续调整
        self.theta = 0.0   # rad
        self.omega = 0.0   # rad/s
    
    #设置卫星初始状态（度、度/秒）
    #参数:把以「度（°）」为单位的角度值，转换成以「弧度（rad）」为单位的角度值
    #theta_deg: 初始角度 (度)
    #omega_deg_s: 初始角速度 (度/秒)，默认为 0
    def set_state(self, theta_deg, omega_deg=0.0):
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)
    
    #欧拉积分更新状态

    #参数:
    #torque: 控制力矩 (N·m)
    #dt: 时间步长 (s)
    #返回:
    #theta_deg, omega_deg_s: 更新后的角度和角速度（单位：度、度/秒）
    
    #支持外部干扰力矩,默认为0，可以自定义
    def apply_torque(self, torque, dt,disturbance=0.0):
        # 总力矩 = 控制力矩 + 干扰力矩
        total_torque = torque + disturbance   
        #角加速=控制力矩/转动惯量
        alpha = total_torque / self.I
        #角速度2=角速度1+角加速度*dt
        self.omega += alpha * dt
        #角度2=角度1+角速度*dt
        self.theta += self.omega * dt
    
    #功能：返回当前角度，单位度°
    def get_angle_deg(self):
        return np.degrees(self.theta)
    
    #功能：返回角速度（度/秒）
    def get_omega_deg(self):
        return np.degrees(self.omega)