"""
单轴卫星动力学系统辨识（satellite_ai 新增模块）

ds27: 新增系统辨识 —— 用随机力矩激励单轴卫星（双积分器），采集输入-输出数据，
      用最小二乘（线性回归，机器学习的最基本方法）拟合离散线性状态空间模型
      x_{k+1} = A x_k + B u_k，作为 MPC 的预测模型。
      注：单轴是线性双积分器，最小二乘可精确恢复真实 A、B；对非线性系统则需用神经网络辨识。

用法（在 satellite_ai 目录下运行）：
  python mpc/system_id.py
"""

import os
import sys

# 将 satellite_ai 根目录加入 sys.path，使 common/single_axis/three_axis 子包可被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from single_axis.sat_env import Satellite

# 与单轴环境一致的物理参数
I = 1.0
DT = 0.02
MAX_TORQUE = 2.0


def generate_data(steps=2000, seed=0):
    """用随机力矩激励卫星，采集 (x_k, u_k, x_{k+1}) 样本。"""
    rng = np.random.default_rng(seed)
    sat = Satellite(I=I)
    sat.set_state(np.radians(rng.uniform(-30.0, 30.0)), 0.0)

    X = []   # 当前状态 [theta, omega]
    U = []   # 输入力矩 u
    Y = []   # 下一状态 [theta, omega]
    for _ in range(steps):
        u = float(rng.uniform(-MAX_TORQUE, MAX_TORQUE))  # 随机激励
        x = np.array([sat.theta, sat.omega])
        sat.apply_torque(u, DT)
        x_next = np.array([sat.theta, sat.omega])
        X.append(x)
        U.append(u)
        Y.append(x_next)
    return (np.array(X), np.array(U).reshape(-1, 1), np.array(Y))


def identify(X, U, Y):
    """最小二乘拟合 x_{k+1} = A x_k + B u_k，返回 (A, B)。"""
    Z = np.hstack([X, U])                    # (N, 3) = [x_k, u_k]
    P = np.linalg.lstsq(Z, Y, rcond=None)[0]  # (3, 2)，解 Z P = Y
    P = P.T                                   # (2, 3) = [A B]
    A = P[:, :2]
    B = P[:, 2:]
    return A, B


if __name__ == "__main__":
    X, U, Y = generate_data()
    A, B = identify(X, U, Y)
    # 注意：Satellite.apply_torque 是「半隐式欧拉」（先更新 ω，再用新 ω 更新 θ），
    # 所以 θ_{k+1} 含 (dt²/I)·u_k 项，真实 B = [[dt²/I],[dt/I]]（而非显式欧拉的 [[0],[dt/I]]）。
    A_true = np.array([[1.0, DT], [0.0, 1.0]])
    B_true = np.array([[DT * DT / I], [DT / I]])

    print("[ds] 辨识出的 A =\n", np.round(A, 6))
    print("[ds] 真实      A =\n", A_true)
    print("[ds] 辨识出的 B =\n", np.round(B, 6))
    print("[ds] 真实      B =\n", B_true)
    err = np.max(np.abs(np.hstack([A, B]) - np.hstack([A_true, B_true])))
    print(f"[ds] 最大辨识误差 = {err:.2e}（应为 ~0，说明最小二乘精确恢复了双积分器模型）")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "identified_model.npz")
    np.savez(out, A=A, B=B, A_true=A_true, B_true=B_true)
    print(f"[ds] 已保存辨识结果 -> {out}")
