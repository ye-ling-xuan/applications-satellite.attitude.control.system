# ppo_gui_autotune.py
# ============================================
# PPO 卫星控制 GUI + 自动调参（Auto Tune）
# ============================================

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stable_baselines3 import PPO
from sat_env import SatelliteAttitudeEnv


# ======================== 环境 ========================
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


# ======================== 测试 ========================
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


# ======================== 性能评估 ========================
def evaluate_performance(time, angle, torque):
    idx = np.where(np.abs(angle) < 1.0)[0]
    settle_time = time[idx[0]] if len(idx) > 0 else 999

    overshoot = np.max(np.abs(angle))
    energy = np.sum(torque**2)

    score = settle_time + 0.1 * overshoot + 0.01 * energy
    return score


# ======================== GUI ========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PPO Auto-Tune GUI")

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

        # reward 参数
        add_slider("w_theta", 1, 10, 5)
        add_slider("w_omega", 0, 5, 1)
        add_slider("w_torque", 0, 3, 0.5)
        add_slider("w_damping", 0, 5, 2)

        # PPO 参数
        add_slider("learning_rate", 1e-5, 1e-3, 1e-4)
        add_slider("gamma", 0.9, 0.999, 0.98)

        # 初始条件
        add_slider("theta0", 0, 60, 30)
        add_slider("omega0", -20, 20, 0)

        # 按钮
        tk.Button(panel, text="Train & Run", command=self.train).pack(pady=5)
        tk.Button(panel, text="Auto Tune 🤖", command=self.auto_tune).pack(pady=5)

        # ===== 图像 =====
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
            n_steps=512,
            batch_size=64,
            verbose=0,
        )

        model.learn(total_timesteps=100000)

        t, angle, torque = run_test(model, env, theta0, omega0)

        # 绘图
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

    # ========================
    def auto_tune(self):
        print("Auto tuning...")

        best_score = float("inf")
        best_config = None

        trials = 6   # 调大更准，但更慢

        for i in range(trials):
            config = {
                "w_theta": np.random.uniform(3, 8),
                "w_omega": np.random.uniform(0.5, 3),
                "w_torque": np.random.uniform(0.1, 2),
                "w_damping": np.random.uniform(0, 4),
                "learning_rate": np.random.uniform(5e-5, 3e-4),
                "gamma": np.random.uniform(0.95, 0.995),
            }

            env = TunableEnv(config)

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                n_steps=512,
                batch_size=64,
                verbose=0,
            )

            model.learn(total_timesteps=50000)

            t, angle, torque = run_test(
                model,
                env,
                self.controls["theta0"].get(),
                self.controls["omega0"].get()
            )

            score = evaluate_performance(t, angle, torque)
            print(f"Trial {i}: score={score:.2f}")

            if score < best_score:
                best_score = score
                best_config = config

        print("Best score:", best_score)

        # 更新滑块
        for k in best_config:
            if k in self.controls:
                self.controls[k].set(best_config[k])

        print("Best config applied!")

        # 用最优参数再训练一次
        self.train()


# ======================== 主程序 ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()