# sat_env.py
import gymnasium as gym
import numpy as np

# ======================== 卫星动力学模型 ========================
class Satellite:
    def __init__(self, I=1.0):
        self.I = I
        self.theta = 0.0   # rad
        self.omega = 0.0   # rad/s

    def set_state(self, theta_deg, omega_deg=0.0):
        self.theta = np.radians(theta_deg)
        self.omega = np.radians(omega_deg)

    def apply_torque(self, torque, dt):
        alpha = torque / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt

    def get_angle_deg(self):
        return np.degrees(self.theta)

    def get_omega_deg(self):
        return np.degrees(self.omega)

# ======================== Gymnasium 环境 ========================
class SatelliteAttitudeEnv(gym.Env):
    def __init__(self, dt=0.01, max_steps=1000, init_angle_range=[20.0, 40.0]):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.init_angle_range = init_angle_range
        self.sat = Satellite(I=1.0)
        self.current_step = 0

        # 动作空间：力矩 [-2.0, 2.0] Nm
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )
        # 观测空间：归一化后的 [theta/pi, omega/10]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -2.0]),
            high=np.array([1.0, 2.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                    # 设置随机种子
        angle = np.random.uniform(*self.init_angle_range)
        self.sat.set_state(angle, 0.0)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        theta_norm = self.sat.theta / np.pi
        omega_norm = self.sat.omega / 10.0
        return np.array([theta_norm, omega_norm], dtype=np.float32)

    def step(self, action):
        torque = np.clip(action[0], -2.0, 2.0)
        self.sat.apply_torque(torque, self.dt)
        self.current_step += 1

        theta = self.sat.theta
        omega = self.sat.omega
        error = -theta                   # 目标角度为 0
        raw_reward = -(10.0 * error**2 + 0.5 * omega**2 + 0.1 * torque**2)
        reward = raw_reward / 10.0

        terminated = (self.current_step >= self.max_steps)
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass