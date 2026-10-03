"""
三轴姿态收敛可视化动画（satellite_ai 新增模块）

ds30: 新增可视化动画 —— 用 matplotlib 3D 动画展示三轴卫星从初始姿态收敛到目标
      姿态（单位四元数）的过程：左图实时画出卫星本体坐标系（红X/绿Y/蓝Z）的旋转，
      右图画等效转角误差随时间收敛的曲线。分别生成 SAC 与 PID 两个动画便于对比。

用法（在 satellite_ai 目录下运行）：
  python visualization/animate_attitude.py
  可选参数：初始欧拉角（度，roll pitch yaw），默认 0 0 60（单轴偏航 60°）
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
from matplotlib.animation import FuncAnimation
from stable_baselines3 import SAC

from three_axis.sat_env3d import Satellite3DEnv
from three_axis.pid3d import PID3D
from three_axis.quaternion_utils import quaternion_error, euler_to_quaternion


def quat_to_rot(q):
    """单位四元数 [w,x,y,z] → 旋转矩阵 R（body→world）。"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def simulate_sac(model, env, init_euler_deg, seconds=6.0):
    """用 SAC 控制器仿真，返回 (四元数历史, 等效转角误差历史 deg)。"""
    env.reset(seed=0)
    env.sat.set_state(q=euler_to_quaternion(*np.radians(init_euler_deg)), omega=np.zeros(3))
    dt = env.config['dt']
    steps = int(seconds / dt)
    qs, errs = [], []
    for _ in range(steps):
        qs.append(env.sat.q.copy())
        errs.append(np.degrees(env._angle_error(env._error_quaternion())))
        action, _ = model.predict(env._get_obs(), deterministic=True)
        env.step(action)
    return np.array(qs), np.array(errs)


def simulate_pid(env, init_euler_deg, seconds=6.0, Kp=3.0, Ki=0.5, Kd=1.0):
    """用 PID 控制器仿真。"""
    env.reset(seed=0)
    env.sat.set_state(q=euler_to_quaternion(*np.radians(init_euler_deg)), omega=np.zeros(3))
    pid = PID3D(Kp=Kp, Ki=Ki, Kd=Kd, dt=env.config['dt'],
                output_limit=2.0, integral_limit=5.0)
    pid.reset()
    dt = env.config['dt']
    steps = int(seconds / dt)
    qs, errs = [], []
    for _ in range(steps):
        qs.append(env.sat.q.copy())
        errs.append(np.degrees(env._angle_error(env._error_quaternion())))
        torque = pid.compute(quaternion_error(env.config['target_q'], env.sat.q))
        env.sat.update(torque, dt)
    return np.array(qs), np.array(errs)


def make_animation(qs, errs, dt, out_path, title):
    """生成 GIF 动画：左 3D 本体坐标系旋转，右误差收敛曲线。"""
    sub = 5                     # ds30: 抽帧，控制 GIF 体积与时长
    qs = qs[::sub]
    errs = errs[::sub]
    time_arr = np.arange(len(qs)) * dt * sub

    fig = plt.figure(figsize=(10, 4.5))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax3.set_xlim([-1.2, 1.2]); ax3.set_ylim([-1.2, 1.2]); ax3.set_zlim([-1.2, 1.2])
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('卫星本体坐标系')

    # 目标坐标系（灰色虚线，表示零姿态）
    for v in np.eye(3):
        ax3.plot([0, v[0]], [0, v[1]], [0, v[2]], color='gray', linestyle=':', linewidth=1)

    body_lines = [ax3.plot([], [], [], color=c, linewidth=2.5)[0] for c in ['r', 'g', 'b']]

    ax2.set_xlim([0, time_arr[-1]]); ax2.set_ylim([0, max(errs) * 1.15])
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('等效转角误差 (deg)')
    ax2.set_title(title)
    err_line, = ax2.plot([], [], 'b-')

    def update(i):
        R = quat_to_rot(qs[i])
        for k, line in enumerate(body_lines):
            v = R @ np.eye(3)[k]
            line.set_data([0, v[0]], [0, v[1]])
            line.set_3d_properties([0, v[2]])
        err_line.set_data(time_arr[:i + 1], errs[:i + 1])
        return body_lines + [err_line]

    anim = FuncAnimation(fig, update, frames=len(qs), interval=dt * sub * 1000, blit=False)
    anim.save(out_path, writer='pillow', fps=30)
    plt.close(fig)
    print(f"[ds] 已保存动画 -> {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    # 初始姿态（欧拉角，度）：默认单轴偏航 60°，SAC/PID 都能干净收敛
    init = [float(x) for x in sys.argv[1:4]] if len(sys.argv) >= 4 else [0.0, 0.0, 60.0]
    seconds = 6.0

    env = Satellite3DEnv(config={'disturbance': False, 'randomize': False})

    # SAC 动画
    sac_model_path = os.path.join(here, "..", "three_axis", "output", "models_v2", "best_model")
    sac_model_path = os.path.normpath(sac_model_path)
    if os.path.exists(sac_model_path + ".zip"):
        model = SAC.load(sac_model_path)
        qs, errs = simulate_sac(model, env, init, seconds)
        make_animation(qs, errs, env.config['dt'],
                       os.path.join(out_dir, "sac_attitude.gif"), "SAC 姿态误差收敛")
    else:
        print(f"[ds] 未找到 SAC 模型 {sac_model_path}.zip，跳过 SAC 动画")

    # PID 动画
    qs, errs = simulate_pid(env, init, seconds)
    make_animation(qs, errs, env.config['dt'],
                   os.path.join(out_dir, "pid_attitude.gif"), "PID 姿态误差收敛")


if __name__ == "__main__":
    main()
