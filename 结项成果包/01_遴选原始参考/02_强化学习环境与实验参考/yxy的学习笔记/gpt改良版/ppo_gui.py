# ppo_gui.py
# ============================================
# PPO 卫星控制 GUI 调参平台
# ============================================

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stable_baselines3 import PPO
from sat_env import SatelliteAttitudeEnv


# ======================== 可调环境 ========================
class TunableEnv(SatelliteAttitudeEnv):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def step(self, action):
        torque = 2.0 * np.tanh(action[0])

        self.sat.apply_torque(torque, self.dt)
        self.current_step += 1

        theta = self.sat.theta
        omega = self.sat.omega
        c = self.config

        reward = (
            -c["w_theta"] * theta**2
            -c["w_omega"] * omega**2
            -c["w_torque"] * torque**2
            -c["w_damping"] * theta * omega
        )

        if abs(theta) < np.radians(1.0):
            reward += 2.0

        terminated = False

        if abs(theta) > np.radians(90):
            reward -= 50
            terminated = True

        if self.current_step >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {}


# ======================== 测试函数 ========================
def run_test(model, env, theta0, omega0):
    env.reset()
    env.sat.set_state(theta0, omega0)
    obs = env._get_obs()

    t_list, angle_list, torque_list = [], [], []

    for i in range(500):
        t = i * env.dt
        theta = env.sat.get_angle_deg()

        action, _ = model.predict(obs, deterministic=True)
        torque = 2.0 * np.tanh(action[0])

        env.sat.apply_torque(torque, env.dt)
        obs = env._get_obs()

        t_list.append(t)
        angle_list.append(theta)
        torque_list.append(torque)

    return np.array(t_list), np.array(angle_list), np.array(torque_list)


# ======================== GUI ========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PPO Satellite Control GUI")

        self.controls = {}

        # ===== 左侧面板 =====
        panel = tk.Frame(root)
        panel.pack(side=tk.LEFT, padx=10)

        def add_slider(name, frm, to, default):
            tk.Label(panel, text=name).pack()
            var = tk.DoubleVar(value=default)
            slider = tk.Scale(panel, from_=frm, to=to, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=var)
            slider.pack()
            self.controls[name] = var

        # reward
        add_slider("w_theta", 1, 10, 5)
        add_slider("w_omega", 0, 5, 1)
        add_slider("w_torque", 0, 3, 0.5)
        add_slider("w_damping", 0, 5, 2)

        # PPO
        add_slider("learning_rate", 1e-5, 1e-3, 1e-4)
        add_slider("gamma", 0.9, 0.999, 0.98)

        # 初始条件
        add_slider("theta0", 0, 60, 30)
        add_slider("omega0", -20, 20, 0)

        # 按钮
        tk.Button(panel, text="Train & Run", command=self.train).pack(pady=10)

        # ===== 右侧画布 =====
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 6))
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

    # ========================
    def train(self):
        print("Training...")

        config = {
            "w_theta": self.controls["w_theta"].get(),
            "w_omega": self.controls["w_omega"].get(),
            "w_torque": self.controls["w_torque"].get(),
            "w_damping": self.controls["w_damping"].get(),
            "learning_rate": self.controls["learning_rate"].get(),
            "gamma": self.controls["gamma"].get(),
        }

        theta0 = self.controls["theta0"].get()
        omega0 = self.controls["omega0"].get()

        env = TunableEnv(config)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            n_steps=1024,
            batch_size=128,
            verbose=0,
        )

        model.learn(total_timesteps=100_000)

        t, angle, torque = run_test(model, env, theta0, omega0)

        # ===== 绘图 =====
        self.ax1.clear()
        self.ax1.plot(t, angle)
        self.ax1.set_title("Angle")
        self.ax1.grid()

        self.ax2.clear()
        self.ax2.plot(t, torque)
        self.ax2.set_title("Torque")
        self.ax2.grid()

        self.canvas.draw()

        print("Done!")


# ======================== 主程序 ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()