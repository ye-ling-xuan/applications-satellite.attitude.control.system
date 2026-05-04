import gymnasium as gym
from gymnasium import spaces
import numpy as np
from satellite import Satellite   # 你的卫星类

class SatelliteEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, max_steps=500, dt=0.01, 
                 reward_weights=(1.0, 0.1, 0.01),
                 success_reward=1.0):
        super().__init__()
        
        self.sat = Satellite(I=1.0)   # 使用你的卫星类，创建卫星实例，转动惯量 1.0
        self.dt = dt
        self.max_steps = max_steps
        self.target_angle = 0.0   # 目标弧度（0 rad）
        
        # 动作空间：力矩 (N·m), 范围 [-2, 2]
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # 观测空间：角度(rad)，角速度(rad/s)
        high_obs = np.array([np.pi, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        
        """
        奖励参数
        奖励函数使用加权平方和，权重可调节（默认角度权重 1.0，角速度 0.1，力矩 0.01）
        step_count 记录当前回合步数，用于判断是否超时。
        """
        self.w_theta, self.w_omega, self.w_torque = reward_weights
        self.success_reward = success_reward
        self.step_count = 0

    def _normalize_angle(self, theta):
        """
        归一化角度到 [-π, π]
        将任意弧度角度映射到 [-π, π] 区间。
        用于奖励计算和终止判断，但不修改卫星内部状态（保持物理连续性）
        """
        return (theta + np.pi) % (2*np.pi) - np.pi

    def _get_obs(self):
        """返回当前观测（弧度值）"""
        return np.array([self.sat.theta, self.sat.omega], dtype=np.float32)

    def _compute_reward(self, theta, omega, torque):
        """
        奖励为负的加权平方和，鼓励小误差、小角速度、小力矩。
        数值上每一步大约在 -0.5 到 0 之间。
        """
        cost = (self.w_theta * theta**2 +
                self.w_omega * omega**2 +
                self.w_torque * torque**2)
        return -cost

    def reset(self, seed=None, options=None):
        """
        super().reset(seed=seed)：初始化随机数生成器 self.np_random，确保可重复性。
        随机初始化角度（-30°~30°）和角速度（-20°/s~20°/s），
        调用 set_state 将度转换为弧度存入卫星。
        重置步数计数器，返回初始观测和空字典 info
        """
        super().reset(seed=seed)
        # 随机初始化角度（-30° ~ 30°）和角速度（-20°/s ~ 20°/s）
        theta_deg = self.np_random.uniform(-30.0, 30.0)
        omega_deg = self.np_random.uniform(-20.0, 20.0)
        self.sat.set_state(theta_deg, omega_deg)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        """
        动作裁剪：确保力矩在合法范围内（虽然动作空间本身会限制，但做一次安全处理）。
        调用 apply_torque：传入控制力矩和步长，不传第三个参数（disturbance 默认为 0）。如果你要添加干扰，可在这里传入。
        获取状态：theta_raw 是卫星内部可能未归一化的角度（可能 >π），omega 是角速度。
        归一化角度：用 _normalize_angle 将角度映射到 [-π, π]，用于奖励计算和终止判断。
        注意：不修改卫星的 self.sat.theta，因为卫星内部保持连续积分，而评估控制效果时使用归一化角度更合理（避免角度跳变影响奖励）。
        计算奖励：使用归一化后的 theta 和角速度。
        终止条件：当角度和角速度绝对值都小于 0.01 rad（≈0.57°）且 0.01 rad/s 时，认为卫星已稳定，设置 terminated=True 并给予额外奖励。
        超时判断：如果步数达到 max_steps，设置 truncated=True。
        构建观测：使用归一化后的 theta 和原始 omega 作为观测（观测空间要求角度在 ±π 内）。
        返回符合 Gymnasium 标准的五元组。
        """
        # 裁剪动作到合法范围
        torque = np.clip(action[0], -2.0, 2.0)
        # 直接调用 apply_torque，不传 disturbance（默认为0）
        self.sat.apply_torque(torque, self.dt)
        self.step_count += 1

        # 获取当前真实状态（弧度）
        theta_raw = self.sat.theta
        omega = self.sat.omega
        # 归一化角度（仅用于奖励和终止判断，不修改卫星内部状态）
        theta = self._normalize_angle(theta_raw)

        # 计算奖励
        reward = self._compute_reward(theta, omega, torque)

        # 终止条件：角度和角速度都足够小
        terminated = (abs(theta) < 0.01 and abs(omega) < 0.01)
        if terminated:
            reward += self.success_reward

        truncated = self.step_count >= self.max_steps

        # 观测使用原始弧度值（未经归一化也没问题，因为卫星模型本身θ可能超出π，但观测空间已定义为±π，这里最好也归一化）
        # 为了与观测空间一致，将归一化后的角度作为观测的一部分
        obs = np.array([theta, omega], dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info

    """
    简单打印当前状态（注意：这里打印的是卫星内部原始角度，可能大于 360°，但仅作调试用）。
    """
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.step_count}: theta={np.degrees(self.sat.theta):.2f}°, "
                  f"omega={np.degrees(self.sat.omega):.2f}°/s")