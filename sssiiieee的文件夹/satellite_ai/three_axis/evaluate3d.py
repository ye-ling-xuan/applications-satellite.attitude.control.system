"""
三轴评估 + PID vs 强化学习对比脚本（satellite_ai 新增模块）

ds19: 三轴评估 —— 对训练好的 RL 模型做 OOD 泛化测试（不同初始姿态），
      并把三轴姿态误差折算成「等效转角」这个标量，复用 metrics.py 统一评估。

ds20: 三轴 PID vs RL 公平对比 —— 同一动力学(satellite3d.py)、同一初始条件、
      同一力矩限幅(2.0 N·m)、同一采样步长(0.01s)，PID 参数取 PID 侧 main_3d 的
      整定结果(Kp=3, Ki=0.5, Kd=1)，用统一 metrics 输出量化指标。

用法：
  python evaluate3d.py satellite_ai_best_model_3d/sac_satellite3d_final sac
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC

from three_axis.sat_env3d import Satellite3DEnv
from three_axis.pid3d import PID3D
from three_axis.quaternion_utils import euler_to_quaternion, quaternion_error
from common.metrics import compute_metrics, print_table

ALGOS = {"sac": SAC, "ppo": PPO}

# PID 参数与 PID 侧 main_3d.py 一致，保证对比公平
PID_GAINS = dict(Kp=3.0, Ki=0.5, Kd=1.0, output_limit=2.0, integral_limit=5.0)


def run_controller(controller, env, euler_deg, horizon_s=10.0):
    """在固定时长内，从给定初始姿态出发跑一个控制器，返回时间/角度误差/力矩范数。

    controller: callable(env) -> 三轴力矩 (3,)
    """
    q0 = euler_to_quaternion(*np.radians(euler_deg))
    env.sat.set_state(q=q0, omega=np.zeros(3))
    dt = env.config['dt']
    steps = int(horizon_s / dt)
    time = np.zeros(steps)
    angle = np.zeros(steps)      # 等效转角误差 (deg)
    torque_mag = np.zeros(steps)  # 力矩范数 (N·m)
    for i in range(steps):
        time[i] = i * dt
        angle[i] = np.degrees(env._angle_error(env._error_quaternion()))
        torque = controller(env)
        env.sat.update(torque, dt)
        torque_mag[i] = np.linalg.norm(torque)
    return time, angle, torque_mag


def make_sac_controller(model, env):
    def controller(env):
        action, _ = model.predict(env._get_obs(), deterministic=True)
        return env.config['max_torque'] * np.tanh(np.asarray(action, dtype=float))
    return controller


def make_pid_controller(env):
    pid = PID3D(dt=env.config['dt'], **PID_GAINS)
    pid.reset()

    def controller(env):
        error_vec = quaternion_error(env.config['target_q'], env.sat.q)
        return pid.compute(error_vec)
    return controller


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "models", "sac_satellite3d_final")
    algo = sys.argv[2].lower() if len(sys.argv) > 2 else "sac"

    if not os.path.exists(model_path + ".zip"):
        print(f"[ds] 未找到模型 {model_path}.zip，请先运行 train3d.py 训练")
        return
    model = ALGOS[algo].load(model_path)

    # 无干扰、无随机化的确定性环境，只评估控制器本身
    env = Satellite3DEnv(config={'disturbance': False, 'randomize': False})

    # OOD 测试用例：不同初始姿态（欧拉角 deg）
    test_cases = [
        (0.0, 0.0, 30.0),     # 单轴偏航 30°
        (30.0, 0.0, 0.0),     # 单轴滚转 30°
        (0.0, 30.0, 0.0),     # 单轴俯仰 30°
        (45.0, 45.0, 45.0),   # 三轴同时 45°
        (60.0, -60.0, 60.0),  # 三轴混合大角度
    ]

    sac_ctl = make_sac_controller(model, env)
    pid_ctl = make_pid_controller(env)

    rows = []
    for euler in test_cases:
        label = "/".join(f"{e:.0f}" for e in euler)
        t_s, a_s, q_s = run_controller(sac_ctl, env, euler)
        t_p, a_p, q_p = run_controller(pid_ctl, env, euler)
        rows.append((f"SAC {label}", compute_metrics(t_s, a_s, torque=q_s)))
        rows.append((f"PID {label}", compute_metrics(t_p, a_p, torque=q_p)))

    print_table(rows)

    # 画一组对比曲线（最后一个用例）
    euler = test_cases[-1]
    t_s, a_s, _ = run_controller(sac_ctl, env, euler)
    t_p, a_p, _ = run_controller(pid_ctl, env, euler)
    plt.figure(figsize=(8, 4))
    plt.plot(t_s, a_s, label="SAC")
    plt.plot(t_p, a_p, label="PID")
    plt.axhline(0, color="gray", linestyle=":")
    plt.xlabel("Time (s)")
    plt.ylabel("Attitude error (deg)")
    plt.title(f"3D attitude control: init euler {euler}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    out_png = f"{algo}_vs_pid_3d.png"
    plt.savefig(out_png, dpi=150)
    print(f"[ds] 已保存对比图 -> {out_png}")


if __name__ == "__main__":
    main()
