import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatelliteEnv(gym.Env):
    """
    单轴卫星姿态控制环境
    状态: [角度(rad), 角速度(rad/s)]
    动作: [控制力矩(N·m)]，连续值
    奖励: 负的加权平方和（角度误差+角速度+控制能耗），稳定到目标时给成功奖励
    """

    def __init__(self, max_steps=500, dt=0.01,
                 disturbance_scale=0.005, enable_disturbance=True,
                 success_bonus=1.0):
        super().__init__()

        # 物理参数
        self.I = 1.0                # 转动惯量
        self.max_torque = 2.0       # 最大力矩
        self.dt = dt
        self.max_steps = max_steps
        self.target_angle = 0.0     # 目标角度（弧度）

        # 干扰：可配置（默认极弱，真实但不破坏训练）
        self.disturbance_scale = disturbance_scale
        self.enable_disturbance = enable_disturbance

        # 成功到达目标的奖励（稳定终止时给出）
        self.success_bonus = success_bonus

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

    def _get_obs(self):
        """当前观测：[角度, 角速度]"""
        return np.array([self.theta, self.omega], dtype=np.float32)

    def set_state(self, theta, omega=0.0):
        """设置指定状态（角度弧度、角速度弧度/秒），供评测指定初始偏角。"""
        self.theta = float(theta)
        self.omega = float(omega)
        self.step_count = 0
        return self._get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if 'theta' in options:
            # 指定初始角度（弧度），用于评测指定偏角
            self.theta = float(options['theta'])
            self.omega = float(options.get('omega', 0.0))
        else:
            # 随机初始化：角度 ±0.5 rad (~30度)，角速度 ±0.2 rad/s
            self.theta = self.np_random.uniform(-0.5, 0.5)
            self.omega = self.np_random.uniform(-0.2, 0.2)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        torque = np.clip(action[0], -self.max_torque, self.max_torque)

        # 微小干扰，真实但不破坏训练
        disturbance = 0.0
        if self.enable_disturbance:
            disturbance = self.np_random.uniform(-self.disturbance_scale, self.disturbance_scale)

        # 动力学
        alpha = (torque + disturbance) / self.I
        self.omega += alpha * self.dt
        self.theta += self.omega * self.dt

        # 角度归一化
        error = self.theta - self.target_angle
        error = (error + np.pi) % (2 * np.pi) - np.pi
        self.theta = self.target_angle + error

        self.step_count += 1

        # 奖励
        reward = self._compute_reward(error, self.omega, torque)

        # 结束条件
        terminated = bool(abs(error) < 0.02 and abs(self.omega) < 0.02)
        truncated = self.step_count >= self.max_steps

        # 成功奖励：稳定到目标附近时额外给一个正奖励
        if terminated:
            reward += self.success_bonus

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self, error, omega, torque):
        """
        最终稳定奖励函数：带干扰也能平滑曲线
        """
        # 权重：角度误差、角速度、控制能耗
        w_angle = 1.0
        w_omega = 0.1
        w_torque = 0.001

        # 核心：使用最短路径角度误差
        cost = (w_angle * (error ** 2) +
                w_omega * (omega ** 2) +
                w_torque * (torque ** 2))

        reward = -cost
        return reward

    def render(self):
        # 简单打印当前状态
        print(f"Step: {self.step_count}, theta={np.degrees(self.theta):.2f}°, omega={np.degrees(self.omega):.2f}°/s")
