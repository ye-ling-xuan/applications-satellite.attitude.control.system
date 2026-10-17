"""仅从已保存的固定轨迹重新绘图；没有新训练或仿真。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with np.load(HERE / "固定轨迹/PID_seed_0_test_00_001.npz", allow_pickle=False) as data:
    trace = {key: data[key].copy() for key in ("t", "error_deg", "omega_deg_s", "torque")}

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
fig.suptitle("单轴PID固定演示：初始45°、3°/s，J=1 kg·m²\n已有轨迹 test_00_001；单条示例不代表总体统计")
panels = [("error_deg", "姿态误差 / °", 1.0),
          ("omega_deg_s", "角速度 / °/s", 0.5),
          ("torque", "实际控制力矩 / N·m", None)]
for ax, (key, label, tolerance) in zip(axes, panels):
    ax.plot(trace["t"], trace[key], color="#2166ac", linewidth=1.8)
    if tolerance is not None:
        ax.axhspan(-tolerance, tolerance, color="#a6bddb", alpha=0.3,
                   label=f"允许范围 ±{tolerance:g}")
        ax.legend(loc="upper right")
    else:
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
        ax.axhline(-0.5, color="#777777", linestyle=":", linewidth=1)
    ax.axvline(4.3, color="#b35806", linestyle="--", linewidth=1)
    ax.grid(alpha=0.2)
    ax.set_ylabel(label)
axes[0].annotate("4.30 s 后持续达标", xy=(4.3, 0), xytext=(7, 18),
                 arrowprops={"arrowstyle": "->", "color": "#b35806"})
axes[-1].set_xlabel("时间 / s")
fig.savefig(HERE.parent / "核心图_PID固定响应.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
