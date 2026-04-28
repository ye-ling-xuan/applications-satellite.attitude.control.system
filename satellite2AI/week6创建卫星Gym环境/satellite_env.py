import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatelliteEnv(gym.Env):
    """
    单轴卫星姿态控制环境
    状态: [角度(rad), 角速度(rad/s)]
    动作: [控制力矩(N·m)]，连续值
    奖励: 负的加权平方和（角度误差+角速度+控制能耗）
    """
    
    def __init__(self, max_steps=500, dt=0.01):
        super().__init__()
        
        # 物理参数
        self.I = 1.0                # 转动惯量
        self.max_torque = 2.0       # 最大力矩
        self.dt = dt
        self.max_steps = max_steps
        self.target_angle = 0.0     # 目标角度（弧度）
        
        # 动作空间：连续力矩，范围 [-max_torque, max_torque]
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32
        )
        
        # 状态空间：角度和角速度
        # 角度范围 [-π, π]，角速度范围设为较大值，但实际会裁剪
        high = np.array([np.pi, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        
        # 内部状态
        self.theta = 0.0
        self.omega = 0.0
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始化：角度 ±0.5 rad (~30度)，角速度 ±0.2 rad/s
        self.theta = self.np_random.uniform(-0.5, 0.5)
        self.omega = self.np_random.uniform(-0.2, 0.2)
        self.step_count = 0
        return np.array([self.theta, self.omega], dtype=np.float32), {}
    
    def step(self, action):
        # 提取动作并限制
        torque = np.clip(action[0], -self.max_torque, self.max_torque)
        
        # 动力学更新（欧拉积分）
        alpha = torque / self.I
        self.omega += alpha * self.dt
        self.theta += self.omega * self.dt
        
        # 角度归一化到 [-π, π]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        
        self.step_count += 1
        
        # 计算奖励
        reward = self._compute_reward(self.theta, self.omega, torque)
        
        # 判断是否结束
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # 可选：如果角度和角速度都很小，视为成功提前结束
        if abs(self.theta) < 0.01 and abs(self.omega) < 0.01:
            terminated = True
        
        # 新状态
        obs = np.array([self.theta, self.omega], dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, theta, omega, torque):
        """
        奖励函数设计（关键！）
        我们希望：角度误差小、角速度小、控制能耗小
        因此奖励可以定义为负的加权平方和
        """
        # 权重可调
        w_angle = 1.0
        w_omega = 0.1
        w_torque = 0.01
        
        cost = (w_angle * theta**2 +
                w_omega * omega**2 +
                w_torque * torque**2)
        reward = -cost
        return reward
    
    def render(self):
        # 简单打印当前状态
        print(f"Step: {self.step_count}, theta={np.degrees(self.theta):.2f}°, omega={np.degrees(self.omega):.2f}°/s")