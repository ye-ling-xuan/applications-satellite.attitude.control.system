import gymnasium as gym #导入Gymnasium库 
from gymnasium import spaces # 导入spaces模块，定义动作和观测空间
import numpy as np # 导入NumPy，用于数值计算

class SatelliteEnv(gym.Env):
    """卫星姿态控制环境
    符合Gymnasium标准接口，可被Stable-Baselines3使用
    """
    
    def __init__(self, config=None):
        super().__init__() # 调用父类gym.Env的初始化方法
        
        # 默认配置
        self.config = {
            'I': 1.0,                    # 转动惯量
            'dt': 0.01,                  # 仿真步长
            'max_steps': 500,            # 最大步数
            'max_torque': 2.0,           # 最大力矩
            'target_angle': 0.0,         # 目标角度(rad)
            'init_angle_range': (-0.5, 0.5),  # 初始角度范围(rad)
            'init_omega_range': (-0.2, 0.2),  # 初始角速度范围(rad/s)
            'disturbance': False,        # 是否添加干扰
            'noise': False,              # 是否添加噪声
        }
        if config: # 如果用户提供了自定义配置
            self.config.update(config) # 用用户配置更新默认配置
        
        # 动作空间：连续力矩
        self.action_space = spaces.Box(
            low=-self.config['max_torque'],
            high=self.config['max_torque'],
            shape=(1,),
            dtype=np.float32
        )
        
        # 状态空间：角度和角速度
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -5.0]),
            high=np.array([np.pi, 5.0]),
            dtype=np.float32
        )
        
        # 内部状态
        self.theta = 0.0   # 角度(rad)
        self.omega = 0.0   # 角速度(rad/s)
        self.step_count = 0
        self.consecutive_stable = 0
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 随机初始化角度和角速度
        low, high = self.config['init_angle_range']
        self.theta = self.np_random.uniform(low, high)
        
        low, high = self.config['init_omega_range']
        self.omega = self.np_random.uniform(low, high)
        
        self.step_count = 0
        self.consecutive_stable = 0
        
        obs = np.array([self.theta, self.omega], dtype=np.float32)
        info = {}
        return obs, info
    
    def step(self, action):
        """执行一步动作，返回下一状态和奖励"""
        # 提取动作并限制
        torque = np.clip(action[0], -self.config['max_torque'], self.config['max_torque'])
        
        # ===== 添加干扰（可选）=====
        if self.config['disturbance']:
            disturbance = self._generate_disturbance()
            torque += disturbance
        
        # ===== 动力学更新 =====
        alpha = torque / self.config['I']
        self.omega += alpha * self.config['dt']
        self.theta += self.omega * self.config['dt']
        
        # 角度归一化到[-π, π]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        
        # ===== 添加测量噪声（可选）=====
        if self.config['noise']:
            measured_theta = self.theta + self.np_random.normal(0, 0.01)
            measured_omega = self.omega + self.np_random.normal(0, 0.02)
        else:
            measured_theta = self.theta
            measured_omega = self.omega
        
        # ===== 计算奖励 =====
        reward = self._compute_reward(measured_theta, measured_omega, torque)
        
        # ===== 判断是否结束 =====
        self.step_count += 1
        
        # 提前终止条件：稳定在目标附近
        terminated = False
        if abs(self.theta - self.config['target_angle']) < 0.02 and abs(self.omega) < 0.02:
            self.consecutive_stable += 1
            if self.consecutive_stable > 10:  # 稳定超过10步
                terminated = True
        else:
            self.consecutive_stable = 0
        
        # 达到最大步数
        truncated = self.step_count >= self.config['max_steps']
        
        # 下一观测值
        obs = np.array([measured_theta, measured_omega], dtype=np.float32)
        info = {'theta': self.theta, 'omega': self.omega}
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, theta, omega, torque):
        """计算奖励函数"""
        # 角度误差（目标0度）
        angle_error = abs(theta - self.config['target_angle'])
        
        # 基础奖励：负的加权平方和
        w_angle = 1.0
        w_omega = 0.1
        w_torque = 0.01
        
        reward = -(w_angle * angle_error**2 + 
                   w_omega * omega**2 + 
                   w_torque * torque**2)
        
        # 稀疏到达奖励
        if angle_error < 0.05 and abs(omega) < 0.05:
            reward += 10.0
        
        # 形状奖励：越接近目标奖励越高
        reward += 0.5 * np.exp(-10 * angle_error)
        
        return reward
    
    def _generate_disturbance(self):
        """生成干扰力矩"""
        # 正弦干扰 + 随机噪声
        t = self.step_count * self.config['dt']
        periodic = 0.05 * np.sin(2 * np.pi * 0.5 * t)
        random = self.np_random.normal(0, 0.01)
        return periodic + random
    
    def render(self):
        """简单打印当前状态"""
        print(f"Step {self.step_count}: θ={np.degrees(self.theta):.2f}°, ω={np.degrees(self.omega):.2f}°/s")