"""
三轴卫星姿态控制强化学习环境（satellite_ai 新增模块）

ds14: 新增三轴 Gym 环境 —— 把三轴刚体卫星姿态控制包装成标准 RL 环境。
      相比单轴（1 自由度双积分器，可解析求解），三轴具有耦合项 ω×(Iω) 与
      四元数运动学的非线性，这才是强化学习相对 PID 更能体现优势的问题。

  状态(7 维) = 误差四元数 q_err(4) + 归一化角速度(3)，均在 [-1,1] 附近
  动作(3 维) = 三轴力矩，归一化 [-1,1] → 实际力矩 max_torque * tanh(a)
  动力学参数（max_torque=2.0、dt=0.01、惯量 diag[1.2,1.0,0.8]）与 PID 三轴侧一致。

ds15: 三轴奖励函数 —— 主误差用「误差四元数的等效转角」2·arccos(|q_err_w|)，
      加角速度范数、力矩能耗、时间惩罚、成功终局奖励、发散惩罚。
ds16: 三轴域随机化 —— 随机化惯量/初始姿态/初始角速度/干扰幅值，增强泛化。
ds17: 终止语义沿用 ds1 规范（超时 truncated，成功/发散 terminated）。
ds21: 奖励改用【线性】角度/角速度惩罚（恒定梯度，解决二次惩罚接近目标时梯度消失）。
ds22: 课程学习 —— set_init_range() 支持训练中逐步增大初始姿态范围（从易到难）。
ds23: potential-based shaping —— 李雅普诺夫势函数 Φ = -(w_sa·e² + w_sw·‖ω‖²)，
      奖励加入 γ·Φ(s')-Φ(s)，理论保证不改变最优策略，但提供稠密梯度。

依赖：satellite3d.py（动力学）、quaternion_utils.py（四元数工具）。
"""

import gymnasium as gym
import numpy as np

from three_axis.satellite3d import Satellite3D
from three_axis.quaternion_utils import euler_to_quaternion, quat_multiply, quat_conjugate


class Satellite3DEnv(gym.Env):
    """三轴卫星姿态控制环境，目标姿态为单位四元数 [1,0,0,0]。"""

    def __init__(self, config=None):
        super().__init__()

        self.config = {
            # 动力学参数（与 PID 三轴侧一致，便于公平对比）
            'I_diag': [1.2, 1.0, 0.8],   # 三轴转动惯量（对角）
            'dt': 0.01,                  # 仿真步长，与 PID 侧一致
            'max_steps': 500,            # 单回合最大步数（5 秒）
            'max_torque': 2.0,           # 单轴最大力矩 (N·m)，与 PID 侧一致
            'target_q': np.array([1.0, 0.0, 0.0, 0.0]),

            # 初始姿态/角速度范围（欧拉角 deg，各轴 ±范围）
            'init_euler_range': (60.0, 60.0, 60.0),
            'init_omega_range': (10.0, 10.0, 10.0),

            # ds14: 角速度归一化尺度
            'omega_scale': np.radians(120.0),

            # ds16: 域随机化
            'randomize': True,
            'I_scale_range': (0.8, 1.2),            # 惯量整体缩放范围
            'init_euler_rand': (90.0, 90.0, 90.0),  # 随机化时各轴 ±90°
            'init_omega_rand': (20.0, 20.0, 20.0),
            'disturbance': True,
            'dist_bias_range': (-0.01, 0.01),       # 三轴常值偏置力矩 (N·m)
            'dist_noise_std_range': (0.0, 0.005),   # 高斯噪声标准差 (N·m)

            # ds15/ds21: 奖励权重（ds21 把角度/角速度改为线性惩罚）
            'w_angle': 10.0,      # 等效转角误差【线性】惩罚
            'w_omega': 2.0,       # 角速度范数【线性】惩罚
            'w_torque': 0.05,     # 能耗惩罚（二次）
            'time_penalty': 0.01,
            'success_bonus': 50.0,
            'diverge_penalty': 50.0,

            # ds23: potential-based shaping（李雅普诺夫势函数）
            'gamma': 0.99,           # 与 SAC 折扣因子一致
            'shaping_w_angle': 1.0,  # 势函数角度权重
            'shaping_w_omega': 0.1,  # 势函数角速度权重

            # 收敛/发散判据
            'success_tol_angle': np.radians(1.0),
            'success_tol_omega': np.radians(1.0),
            'success_hold_steps': 10,
            'diverge_angle': np.radians(90.0),
        }
        if config is not None:
            self.config.update(config)

        # ds14: 动作空间 = 三轴归一化力矩；观测空间 = 7 维（误差四元数 + 归一化角速度）
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(7, -1.0, dtype=np.float32),
            high=np.full(7, 1.0, dtype=np.float32),
        )

        self.sat = Satellite3D()
        self.step_count = 0
        self.hold_count = 0
        self.dist_bias = np.zeros(3)
        self.dist_noise_std = 0.0

    # ---------- 误差四元数与等效转角 ----------
    def _error_quaternion(self):
        """q_err = q_target ⊗ q_current⁻¹，并让标量部分非负（q 与 -q 等价）。"""
        q_err = quat_multiply(self.config['target_q'], quat_conjugate(self.sat.q))
        if q_err[0] < 0:
            q_err = -q_err
        return q_err

    def _angle_error(self, q_err):
        """等效转角误差 = 2·arccos(|q_err_w|)，单位 rad，范围 [0, π]。"""
        return 2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))

    # ds22: 课程学习 —— 训练中逐步增大初始姿态范围（从小到大，由易到难）
    def set_init_range(self, max_angle_deg):
        """把初始姿态范围（各轴 ±）调整为 max_angle_deg 度。"""
        r = float(max_angle_deg)
        self.config['init_euler_range'] = (r, r, r)
        self.config['init_euler_rand'] = (r, r, r)

    # ---------- 观测 ----------
    def _get_obs(self):
        q_err = self._error_quaternion()
        omega_norm = self.sat.omega / self.config['omega_scale']
        return np.concatenate([q_err, omega_norm]).astype(np.float32)  # (7,)

    # ---------- reset ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config

        if cfg['randomize']:
            # ds16: 域随机化 —— 每次 reset 随机抽取惯量/初始姿态/角速度/干扰
            scale = self.np_random.uniform(*cfg['I_scale_range'])
            self.sat.I = np.diag(cfg['I_diag']) * scale
            self.sat.I_inv = np.linalg.inv(self.sat.I)
            euler = [self.np_random.uniform(-r, r) for r in cfg['init_euler_rand']]
            omega = [self.np_random.uniform(-r, r) for r in cfg['init_omega_rand']]
            self.dist_bias = self.np_random.uniform(*cfg['dist_bias_range'], size=3)
            self.dist_noise_std = self.np_random.uniform(*cfg['dist_noise_std_range'])
        else:
            self.sat.I = np.diag(cfg['I_diag'])
            self.sat.I_inv = np.linalg.inv(self.sat.I)
            euler = [self.np_random.uniform(-r, r) for r in cfg['init_euler_range']]
            omega = [self.np_random.uniform(-r, r) for r in cfg['init_omega_range']]
            self.dist_bias = np.zeros(3)
            self.dist_noise_std = 0.0

        q0 = euler_to_quaternion(*[np.radians(e) for e in euler])
        self.sat.set_state(q=q0, omega=np.radians(np.array(omega, dtype=float)))
        self.step_count = 0
        self.hold_count = 0
        return self._get_obs(), {}

    # ---------- step ----------
    def step(self, action):
        cfg = self.config

        # 三轴力矩映射（tanh 平滑限幅）
        torque = cfg['max_torque'] * np.tanh(np.asarray(action, dtype=float))

        # ds23: 保存更新前状态，供 potential-based shaping 计算 Φ(s)
        prev_angle = self._angle_error(self._error_quaternion())
        prev_omega = self.sat.omega.copy()

        # ds16: 三轴干扰 = 常值偏置 + 高斯噪声
        disturbance = np.zeros(3)
        if cfg['disturbance']:
            disturbance = self.dist_bias + self.np_random.normal(0.0, self.dist_noise_std, size=3)

        self.sat.update(torque + disturbance, cfg['dt'])
        self.step_count += 1

        q_err = self._error_quaternion()
        angle_err = self._angle_error(q_err)
        omega = self.sat.omega

        reward = self._compute_reward(prev_angle, prev_omega, angle_err, omega, torque)

        # ds17: 终止语义沿用 ds1（成功/发散 terminated，超时 truncated）
        terminated = False
        truncated = False

        if (angle_err < cfg['success_tol_angle']
                and float(np.linalg.norm(omega)) < cfg['success_tol_omega']):
            self.hold_count += 1
            if self.hold_count >= cfg['success_hold_steps']:
                terminated = True
                reward += cfg['success_bonus']   # ds15: 成功终局奖励
        else:
            self.hold_count = 0

        if angle_err > cfg['diverge_angle']:
            terminated = True
            reward -= cfg['diverge_penalty']

        if self.step_count >= cfg['max_steps']:
            truncated = True

        info = {
            'angle_err': angle_err,          # 等效转角误差 (rad)
            'omega': omega.copy(),
            'torque': torque.copy(),
            'euler_deg': self.sat.get_euler_deg(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ---------- 奖励 ----------
    def _compute_reward(self, prev_angle, prev_omega, angle_err, omega, torque):
        cfg = self.config

        # ds23: potential-based shaping —— 李雅普诺夫势函数 Φ = -(w_sa·e² + w_sw·‖ω‖²)
        #       shaping = γ·Φ(s') - Φ(s)，理论保证不改变最优策略，但提供稠密梯度，
        #       帮助价值函数更好地引导收敛（相比只有线性代价）。
        def potential(e, w):
            return -(cfg['shaping_w_angle'] * e ** 2
                     + cfg['shaping_w_omega'] * float(np.sum(w ** 2)))

        shaping = cfg['gamma'] * potential(angle_err, omega) - potential(prev_angle, prev_omega)

        # ds21: 线性代价（恒定梯度，引导精确收敛）
        cost = (cfg['w_angle'] * angle_err
                + cfg['w_omega'] * float(np.linalg.norm(omega))
                + cfg['w_torque'] * float(np.sum(torque ** 2)))

        return float(shaping - cost - cfg['time_penalty'])

    def render(self):
        pass
