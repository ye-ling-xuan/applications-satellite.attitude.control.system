# sat_env.py
# ============================================
# 卫星姿态控制强化学习环境（改良版）
# 特点：
# 1. 防发散终止机制
# 2. 更合理的奖励函数（含阻尼项）
# 3. 动作平滑限制（tanh）
# 4. 状态归一化基于物理范围
# ============================================

import gymnasium as gym
import numpy as np

# ======================== 卫星动力学模型 ========================
class Satellite:
    def __init__(self, I=1.0):
        """
        I: 转动惯量
        """
        self.I = I
        self.theta = 0.0   # rad
        self.omega = 0.0   # rad/s

    def set_state(self, theta_deg, omega_deg=0.0):
        """设置初始状态（角度单位：度）"""
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)

    def apply_torque(self, torque, dt):
        """动力学更新"""
        alpha = torque / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt

    def get_angle_deg(self):
        return np.degrees(self.theta)

    def get_omega_deg(self):
        return np.degrees(self.omega)


# ======================== Gym 环境 ========================
class SatelliteAttitudeEnv(gym.Env):
    def __init__(self, dt=0.02, max_steps=500):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.sat = Satellite(I=1.0)

        self.current_step = 0

        # ===== 动作空间 =====
        # 输出范围 [-1, 1]，再映射为实际力矩
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ===== 状态空间 =====
        # 归一化范围 [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # ===== 物理归一化尺度 =====
        self.theta_scale = np.radians(40.0)   # 最大约 40°
        self.omega_scale = np.radians(50.0)   # 最大角速度

    # ======================== reset ========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始角度：20°~40°
        init_angle = np.random.uniform(20.0, 40.0)
        self.sat.set_state(init_angle, 0.0)

        self.current_step = 0
        return self._get_obs(), {}

    # ======================== 状态 ========================
    def _get_obs(self):
        theta_norm = self.sat.theta / self.theta_scale
        omega_norm = self.sat.omega / self.omega_scale

        return np.array([theta_norm, omega_norm], dtype=np.float32)

    # ======================== step ========================
    def step(self, action):
        # ===== 动作映射（平滑限制）=====
        torque = 2.0 * np.tanh(action[0])   # 比 clip 更稳定

        # ===== 系统推进 =====
        self.sat.apply_torque(torque, self.dt)
        self.current_step += 1

        theta = self.sat.theta
        omega = self.sat.omega

        # ===== 奖励函数（关键）=====
        # 1. 角度误差（主目标）
        # 2. 角速度（阻尼）
        # 3. 控制能量
        # 4. 阻尼引导项（theta * omega）
        reward = (
            -5.0 * theta**2
            -1.0 * omega**2
            -0.5 * torque**2
            -2.0 * theta * omega
        )

        # ===== 接近目标奖励 =====
        if abs(theta) < np.radians(1.0):
            reward += 2.0

        terminated = False

        # ===== 发散惩罚 =====
        if abs(theta) > np.radians(90):
            reward -= 50.0
            terminated = True

        # ===== 时间终止 =====
        if self.current_step >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        pass