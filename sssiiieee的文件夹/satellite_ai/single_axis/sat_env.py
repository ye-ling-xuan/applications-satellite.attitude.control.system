"""
单轴卫星姿态控制强化学习环境（satellite_ai 改进版）

说明：
  本文件是从原项目强化学习代码复制后改进的版本，原代码未做任何修改。
  原代码来源（合并了各版本的优点）：
    - yxy的学习笔记/gpt改良版/sat_env.py（动作平滑 tanh、观测归一化、发散终止）
    - cms的代码批注/satellite_env.py（config 配置化、干扰/噪声开关、稳定计数）
    - satellite2AI/week6创建卫星Gym环境/satellite_env.py（Gymnasium 标准接口）

改进点汇总（均以 "ds:" 注释标注，便于追溯）：
  ds1: 修正 episode 终止语义 —— 超时用 truncated，不再误用 terminated
  ds2: 奖励函数系统化 —— 去手工交叉项，新增成功终局奖励 + 时间惩罚
  ds3: 干扰模型更真实 —— 常值偏置 + 轨道周期正弦 + 高斯噪声
  ds4: 域随机化(domain randomization) —— 随机化惯量/初始状态/干扰幅值
  ds5: 观测归一化统一到 [-1,1]，并在 info 中保留真实物理量供评估使用
"""

import gymnasium as gym
import numpy as np


class Satellite:
    """单轴刚体卫星动力学模型（双积分器：theta'' = torque / I）

    保留原实现。为便于域随机化（ds4），转动惯量 I 可在 reset 时更新。
    """

    def __init__(self, I=1.0):
        self.I = I
        self.theta = 0.0   # rad
        self.omega = 0.0   # rad/s

    def set_state(self, theta_rad, omega_rad=0.0):
        self.theta = theta_rad
        self.omega = omega_rad

    def apply_torque(self, torque, dt):
        alpha = torque / self.I
        self.omega += alpha * dt
        self.theta += self.omega * dt


class SatelliteAttitudeEnv(gym.Env):
    """单轴卫星姿态控制环境（改进版）

    state  : [theta_norm, omega_norm] ∈ [-1, 1]   （ds5 观测已归一化）
    action : 归一化力矩 a ∈ [-1, 1]，实际力矩 = max_torque * tanh(a)
    reward : 见 _compute_reward（ds2 重构）
    """

    def __init__(self, config=None):
        super().__init__()

        # ---------- 默认配置 ----------
        self.config = {
            'I': 1.0,                 # 转动惯量 (kg·m²)
            'dt': 0.02,               # 仿真步长 (s)
            'max_steps': 500,         # 单回合最大步数
            'max_torque': 2.0,        # 最大力矩 (N·m)
            'target_angle': 0.0,      # 目标角度 (rad)

            # 初始状态范围（rad / rad/s）
            'init_angle_range': (np.radians(20.0), np.radians(40.0)),
            'init_omega_range': (np.radians(-10.0), np.radians(10.0)),

            # ds5: 观测归一化尺度（观测值 / scale ∈ [-1, 1]）
            'theta_scale': np.radians(90.0),   # 角度上限
            'omega_scale': np.radians(60.0),   # 角速度上限

            # ds3: 干扰模型参数（偏置 + 周期正弦 + 高斯噪声）
            'disturbance': True,
            'dist_bias': 0.0,             # 常值偏置力矩 (N·m)
            'dist_periodic_amp': 0.0,     # 周期项幅值 (N·m)
            'dist_periodic_freq': 0.5,    # 周期项频率 (Hz)
            'dist_noise_std': 0.0,        # 高斯噪声标准差 (N·m)

            # ds2: 奖励权重（清晰的三项二次代价 + 时间惩罚 + 成功奖励）
            'w_theta': 10.0,       # 角度误差二次惩罚
            'w_omega': 0.5,        # 角速度二次惩罚
            'w_torque': 0.05,      # 控制能耗惩罚
            'time_penalty': 0.01,  # 每步时间惩罚，防止"磨蹭式"收敛
            'success_bonus': 20.0, # 成功收敛的终局奖励
            'diverge_penalty': 50.0,  # 发散惩罚

            # ds2: 收敛判据（需持续 N 步才算成功，避免偶然进入目标带）
            'success_tol_theta': np.radians(1.0),
            'success_tol_omega': np.radians(1.0),
            'success_hold_steps': 10,
            'diverge_theta': np.radians(90.0),

            # ds4: 域随机化开关与范围
            'randomize': True,
            'I_range': (0.8, 1.2),
            'dist_bias_range': (-0.01, 0.01),
            'dist_periodic_amp_range': (0.0, 0.05),
            'dist_noise_std_range': (0.0, 0.01),
        }
        if config is not None:
            self.config.update(config)

        # 动作空间：归一化力矩 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # ds5: 观测空间统一为 [-1, 1]（环境内部已归一化）
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.sat = Satellite(I=self.config['I'])
        self.step_count = 0
        self.hold_count = 0   # ds2: 连续稳定计数

        # ds3/ds4: 实际生效的干扰参数（reset 时由域随机化采样）
        self.dist_bias = self.config['dist_bias']
        self.dist_periodic_amp = self.config['dist_periodic_amp']
        self.dist_noise_std = self.config['dist_noise_std']

    # ---------------- reset ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        cfg = self.config

        # ds4: 域随机化 —— 每次 reset 随机抽取惯量/初始状态/干扰幅值，
        #      训练出的策略对不同参数具有更强的泛化能力
        if cfg['randomize']:
            self.sat.I = self.np_random.uniform(*cfg['I_range'])
            init_angle = self.np_random.uniform(*cfg['init_angle_range'])
            init_omega = self.np_random.uniform(*cfg['init_omega_range'])
            self.dist_bias = self.np_random.uniform(*cfg['dist_bias_range'])
            self.dist_periodic_amp = self.np_random.uniform(*cfg['dist_periodic_amp_range'])
            self.dist_noise_std = self.np_random.uniform(*cfg['dist_noise_std_range'])
        else:
            self.sat.I = cfg['I']
            init_angle = self.np_random.uniform(*cfg['init_angle_range'])
            init_omega = self.np_random.uniform(*cfg['init_omega_range'])
            self.dist_bias = cfg['dist_bias']
            self.dist_periodic_amp = cfg['dist_periodic_amp']
            self.dist_noise_std = cfg['dist_noise_std']

        self.sat.set_state(init_angle, init_omega)
        self.step_count = 0
        self.hold_count = 0
        return self._get_obs(), {}

    # ---------------- 观测 ----------------
    def _get_obs(self):
        # ds5: 统一归一化到 [-1, 1]，避免原始量纲差异导致训练不稳定
        theta_norm = self.sat.theta / self.config['theta_scale']
        omega_norm = self.sat.omega / self.config['omega_scale']
        return np.array([theta_norm, omega_norm], dtype=np.float32)

    # ---------------- step ----------------
    def step(self, action):
        cfg = self.config

        # 动作平滑映射：tanh 比硬 clip 更利于梯度稳定（保留原改良版做法）
        torque = cfg['max_torque'] * np.tanh(float(action[0]))

        # ds3: 更真实的干扰 = 常值偏置 + 轨道周期正弦 + 高斯噪声
        #      （原实现只有幅值极小的均匀白噪声）
        disturbance = 0.0
        if cfg['disturbance']:
            t = self.step_count * cfg['dt']
            disturbance = (
                self.dist_bias
                + self.dist_periodic_amp * np.sin(2 * np.pi * cfg['dist_periodic_freq'] * t)
                + self.np_random.normal(0.0, self.dist_noise_std)
            )

        # 动力学推进
        self.sat.apply_torque(torque + disturbance, cfg['dt'])
        self.step_count += 1

        theta = self.sat.theta
        omega = self.sat.omega

        # dense 奖励（二次代价 + 时间惩罚）
        reward = self._compute_reward(theta, omega, torque)

        # ds1: 终止语义修正 —— 成功/发散用 terminated，超时用 truncated。
        #      原实现把超时也设为 terminated，会让价值函数在 timeout 处
        #      被错误截断 bootstrap，拖慢收敛甚至导致发散。
        terminated = False
        truncated = False

        # 成功：持续 hold 步满足阈值 → 提前终止，并给终局奖励（ds2）
        if (abs(theta - cfg['target_angle']) < cfg['success_tol_theta']
                and abs(omega) < cfg['success_tol_omega']):
            self.hold_count += 1
            if self.hold_count >= cfg['success_hold_steps']:
                terminated = True
                reward += cfg['success_bonus']   # ds2: 成功终局奖励
        else:
            self.hold_count = 0

        # 发散：角度超过阈值
        if abs(theta) > cfg['diverge_theta']:
            terminated = True
            reward -= cfg['diverge_penalty']

        # ds1: 超时 → truncated（而非 terminated）
        if self.step_count >= cfg['max_steps']:
            truncated = True

        # ds5: info 中保留真实物理量，供评估/绘图使用（观测已归一化，不能直接用于物理指标）
        info = {
            'theta': theta,
            'omega': omega,
            'torque': torque,
            'disturbance': disturbance,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ---------------- 奖励 ----------------
    def _compute_reward(self, theta, omega, torque):
        cfg = self.config

        # ds2: 去掉了原改良版里手工调的交叉项 -2*theta*omega（该交叉项
        #      在不同象限会变相奖励/惩罚某些状态，权重偏大易诱发振荡），
        #      改为清晰、可解释的三项二次代价。
        e = theta - cfg['target_angle']
        cost = (cfg['w_theta'] * e ** 2
                + cfg['w_omega'] * omega ** 2
                + cfg['w_torque'] * torque ** 2)

        # ds2: 每步时间惩罚，鼓励"又快又稳"，避免策略磨蹭式收敛
        reward = -cost - cfg['time_penalty']
        return float(reward)

    def render(self):
        pass
