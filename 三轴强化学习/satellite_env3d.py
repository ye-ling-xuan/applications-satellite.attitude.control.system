"""
三轴卫星姿态控制强化学习环境（自包含，不依赖 pid/ 目录）。

观察空间: [误差旋转向量(3), 角速度(3)]，共 6 维
动作空间: [三轴控制力矩(N·m)]，连续值，每轴范围 [-max_torque, max_torque]
奖励: 负的加权平方和（指向误差 + 角速度 + 控制能耗），稳定到目标时给成功奖励

动力学/运动学方程移植自 pid/three_axis/（已验证）:
    ω̇ = I⁻¹ (τ − ω × (Iω))
    dq/dt = 0.5 · q ⊗ (0, ω)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ---------------- 四元数工具（标量在前 [w, x, y, z]）----------------
def _quat_multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _euler_to_quat(roll, pitch, yaw):
    """欧拉角（弧度，ZYX 顺序）转四元数"""
    cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def _quat_error(q_current, q_target):
    """指向误差 = q_target ⊗ q_current⁻¹，返回轴角误差向量（幅值 = 误差角 rad）"""
    q_err = _quat_multiply(q_target, _quat_conjugate(q_current))
    q_err = q_err / np.linalg.norm(q_err)
    angle = 2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = q_err[1:] / np.linalg.norm(q_err[1:])
    return angle * axis


class SatelliteEnv3D(gym.Env):
    """
    三轴卫星姿态控制环境。
    状态: [误差旋转向量(3), 角速度(3)]
    动作: [控制力矩(3)]，连续值
    奖励: 负的加权平方和（指向误差+角速度+控制能耗）
    """

    def __init__(self, max_steps=500, dt=0.01,
                 max_torque=2.0, inertia=None,
                 enable_disturbance=True, disturbance_scale=0.005,
                 success_bonus=1.0):
        super().__init__()

        # 物理参数
        self.max_torque = max_torque
        self.dt = dt
        self.max_steps = max_steps
        if inertia is None:
            inertia = np.diag([1.2, 1.0, 0.8])
        self.I = np.asarray(inertia, dtype=np.float64)
        self.I_inv = np.linalg.inv(self.I)

        # 干扰（可配置，每轴独立均匀分布）
        self.enable_disturbance = enable_disturbance
        self.disturbance_scale = disturbance_scale

        # 成功奖励
        self.success_bonus = success_bonus

        # 目标姿态：单位四元数 [1,0,0,0]
        self.target_q = np.array([1.0, 0.0, 0.0, 0.0])

        # 动作空间：三轴力矩
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(3,),
            dtype=np.float32
        )

        # 观测空间：[误差向量(3), 角速度(3)]
        # 误差向量各分量 ∈ [-π, π]，角速度范围设较大值
        high = np.array([np.pi, np.pi, np.pi, 5.0, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # 内部状态
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.step_count = 0

    def _get_obs(self):
        error_vec = _quat_error(self.q, self.target_q)
        return np.concatenate([error_vec, self.omega]).astype(np.float32)

    def set_state(self, q=None, omega=None):
        """设置姿态（四元数）和角速度，供评测指定初始偏角。"""
        if q is not None:
            self.q = np.asarray(q, dtype=np.float64)
            self.q = self.q / np.linalg.norm(self.q)
        if omega is not None:
            self.omega = np.asarray(omega, dtype=np.float64).copy()
        self.step_count = 0
        return self._get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if 'q' in options:
            # 指定初始姿态（四元数）
            self.q = np.asarray(options['q'], dtype=np.float64)
            self.q = self.q / np.linalg.norm(self.q)
            self.omega = np.asarray(options.get('omega', np.zeros(3)), dtype=np.float64)
        else:
            # 随机初始化：三轴欧拉角各自随机偏转（真正的三轴机动，含轴间耦合），角速度 ±0.2 rad/s
            euler = self.np_random.uniform(-0.6, 0.6, size=3)  # 每轴 ±0.6 rad ≈ ±34°
            self.q = _euler_to_quat(euler[0], euler[1], euler[2])
            self.q = self.q / np.linalg.norm(self.q)
            self.omega = self.np_random.uniform(-0.2, 0.2, size=3)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        torque = np.clip(np.asarray(action, dtype=np.float64), -self.max_torque, self.max_torque)

        # 微小干扰（每轴独立均匀分布）
        disturbance = np.zeros(3)
        if self.enable_disturbance:
            disturbance = self.np_random.uniform(-self.disturbance_scale, self.disturbance_scale, size=3)

        # 欧拉动力学：ω̇ = I⁻¹ (τ − ω × Iω)
        I_omega = self.I @ self.omega
        omega_dot = self.I_inv @ (torque + disturbance - np.cross(self.omega, I_omega))
        self.omega = self.omega + omega_dot * self.dt

        # 角速度限幅（卫星有最大转速，也避免观测/奖励发散）
        max_omega = 10.0
        omega_norm = float(np.linalg.norm(self.omega))
        if omega_norm > max_omega:
            self.omega = self.omega / omega_norm * max_omega

        # 四元数运动学：dq/dt = 0.5 · q ⊗ (0, ω)
        w, x, y, z = self.q
        wx, wy, wz = self.omega
        q_dot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
             w*wx + y*wz - z*wy,
             w*wy - x*wz + z*wx,
             w*wz + x*wy - y*wx,
        ])
        self.q = self.q + q_dot * self.dt
        self.q = self.q / np.linalg.norm(self.q)

        self.step_count += 1

        # 指向误差（轴角向量）
        error_vec = _quat_error(self.q, self.target_q)
        error_angle = float(np.linalg.norm(error_vec))

        # 奖励
        reward = self._compute_reward(error_vec, self.omega, torque)

        # 结束条件：指向误差 < 0.02 rad 且角速度 < 0.02 rad/s
        terminated = bool(error_angle < 0.02 and float(np.linalg.norm(self.omega)) < 0.02)
        truncated = self.step_count >= self.max_steps

        # 成功奖励
        if terminated:
            reward += self.success_bonus

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self, error_vec, omega, torque):
        w_angle = 1.0
        w_omega = 0.1
        w_torque = 0.001
        # 对误差/角速度代价限幅，避免大角度时奖励过大导致价值函数不稳定
        angle_cost = min(float(np.dot(error_vec, error_vec)), 4.0)   # ≤ 4（≈2 rad 误差）
        omega_cost = min(float(np.dot(omega, omega)), 25.0)          # ≤ 25（≈5 rad/s）
        cost = (w_angle * angle_cost + w_omega * omega_cost +
                w_torque * float(np.dot(torque, torque)))
        return -cost

    def render(self):
        error_angle = float(np.linalg.norm(_quat_error(self.q, self.target_q)))
        print(f"Step: {self.step_count}, pointing error={np.degrees(error_angle):.2f}°, "
              f"omega={np.degrees(self.omega)}")
