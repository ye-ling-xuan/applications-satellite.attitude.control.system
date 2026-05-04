# compare_pid_ppo.py
import numpy as np
import matplotlib.pyplot as plt
from sat_env import Satellite, SatelliteAttitudeEnv
from stable_baselines3 import PPO

# ---------------------- PID 控制器 ----------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target_deg, current_deg):
        target = np.radians(target_deg)
        current = np.radians(current_deg)
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ---------------------- 仿真运行函数 ----------------------
def run_pid(init_angle, dt=0.01, steps=1000):
    sat = Satellite(I=1.0)
    sat.set_state(init_angle, 0.0)
    pid = PIDController(Kp=3.0, Ki=0.5, Kd=1.0, dt=dt)
    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)

    for i in range(steps):
        time[i] = i * dt
        angles[i] = sat.get_angle_deg()
        omegas[i] = sat.get_omega_deg()
        torque = pid.compute(0.0, angles[i])
        torques[i] = torque
        sat.apply_torque(torque, dt)
    return time, angles, omegas, torques

def run_ppo(init_angle, model, dt=0.01, steps=1000):
    env = SatelliteAttitudeEnv()   # 仅用其卫星实例和动力学，不通过reset
    env.sat.set_state(init_angle, 0.0)
    obs = np.array([env.sat.theta / np.pi, env.sat.omega / 10.0], dtype=np.float32)

    time = np.zeros(steps)
    angles = np.zeros(steps)
    omegas = np.zeros(steps)
    torques = np.zeros(steps)

    for i in range(steps):
        time[i] = i * dt
        angles[i] = env.sat.get_angle_deg()
        omegas[i] = env.sat.get_omega_deg()
        action, _ = model.predict(obs, deterministic=True)
        torque = np.clip(action[0], -2.0, 2.0)
        torques[i] = torque
        env.sat.apply_torque(torque, dt)
        obs = np.array([env.sat.theta / np.pi, env.sat.omega / 10.0], dtype=np.float32)
    return time, angles, omegas, torques

# ---------------------- 性能指标计算 ----------------------
def compute_metrics(time, angle, torque, target=0.0, tolerance=0.5):
    """计算超调量(overshoot), 调节时间(settling_time), 稳态误差(steady_state_error), 累计能耗"""
    # 超调量：角度超过目标的最大偏离量（百分比）
    if target == 0.0:
        overshoot = np.max(angle) if np.max(angle) > 0 else 0.0
    else:
        overshoot = max(np.max(angle) - target, 0)

    # 调节时间：进入目标±tolerance以内且不再超出的最早时间
    idx = np.where(np.abs(angle - target) <= tolerance)[0]
    if len(idx) > 0:
        # 找到第一个连续保持的区间
        first = idx[0]
        # 简单取第一个进入容差区的时间作为调节时间
        settling_time = time[first]
    else:
        settling_time = np.inf

    # 稳态误差：最后10%时间内的平均误差绝对值
    steady_portion = int(len(angle) * 0.9)
    steady_error = np.mean(np.abs(angle[steady_portion:] - target))

    # 累计能耗：力矩平方的积分（近似）
    energy = np.sum(torque**2) * (time[1] - time[0])

    return overshoot, settling_time, steady_error, energy

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 加载PPO模型
    try:
        ppo_model = PPO.load("ppo_satellite.zip")
    except:
        print("not found ppo_satellite.zip, please run train_ppo.py first to train and save the model.")
        exit()

    initial_angle = 30.0
    dt = 0.01
    steps = 1000

    # 运行仿真
    t_pid, a_pid, w_pid, trq_pid = run_pid(initial_angle, dt, steps)
    t_ppo, a_ppo, w_ppo, trq_ppo = run_ppo(initial_angle, ppo_model, dt, steps)

    # 计算指标
    pid_metrics = compute_metrics(t_pid, a_pid, trq_pid, target=0.0)
    ppo_metrics = compute_metrics(t_ppo, a_ppo, trq_ppo, target=0.0)

    print("="*60)
    print(f"{'指标':<20} {'PID':<20} {'PPO':<20}")
    print("-"*60)
    print(f"{'超调量 (deg)':<20} {pid_metrics[0]:<20.3f} {ppo_metrics[0]:<20.3f}")
    print(f"{'调节时间 (s)':<20} {pid_metrics[1]:<20.3f} {ppo_metrics[1]:<20.3f}")
    print(f"{'稳态误差 (deg)':<20} {pid_metrics[2]:<20.3f} {ppo_metrics[2]:<20.3f}")
    print(f"{'累计能耗 (Nm²·s)':<20} {pid_metrics[3]:<20.3f} {ppo_metrics[3]:<20.3f}")
    print("="*60)

    # 绘图对比
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_pid, a_pid, 'b-', label='PID', linewidth=2)
    axes[0].plot(t_ppo, a_ppo, 'r--', label='PPO', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle=':')
    axes[0].set_ylabel('Angle (deg)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title(f'Attitude Response from {initial_angle}° to 0°')

    axes[1].plot(t_pid, w_pid, 'b-', label='PID', linewidth=2)
    axes[1].plot(t_ppo, w_ppo, 'r--', label='PPO', linewidth=2)
    axes[1].axhline(0, color='gray', linestyle=':')
    axes[1].set_ylabel('Omega (deg/s)')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t_pid, trq_pid, 'b-', label='PID', linewidth=2)
    axes[2].plot(t_ppo, trq_ppo, 'r--', label='PPO', linewidth=2)
    axes[2].axhline(0, color='gray', linestyle=':')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()