"""
模型预测控制（MPC）控制器（satellite_ai 新增模块）

ds28: 新增 MPC —— 基于辨识出的离散线性模型 x_{k+1} = A x_k + B u_k，
      在有限时域 N 内最小化代价 Σ(xᵀQx + R u²)，用「堆叠预测矩阵」把问题
      化成无约束二次规划，闭式解 U* = -H⁻¹f，取第一个控制量滚动优化
      （receding horizon），并对力矩做饱和限幅。

  - 预测矩阵：X = M·x0 + L·U（X 为未来 N 步状态堆叠，U 为未来 N 步控制堆叠）
  - 代价：J = XᵀQ̄X + UᵀR̄U  →  0.5·UᵀHU + fᵀU + const
  - 无约束最优：U* = -H⁻¹f
"""

import numpy as np


class MPCController:
    def __init__(self, A, B, Q, R, horizon=20, max_torque=2.0):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = float(R)
        self.N = int(horizon)
        self.umax = max_torque
        self._build_prediction()

    def _build_prediction(self):
        n = self.A.shape[0]
        m = self.B.shape[1]
        N = self.N

        # ds28: 堆叠预测矩阵 X = M x0 + L U
        M = np.zeros((n * N, n))
        L = np.zeros((n * N, m * N))
        Apow = np.eye(n)
        for k in range(N):
            Apow = Apow @ self.A            # A^{k+1}
            M[k * n:(k + 1) * n, :] = Apow
        for i in range(N):                  # 未来状态 x_{i+1}
            for j in range(i + 1):          # 受 u_j 影响（j <= i）
                L[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(self.A, i - j) @ self.B

        Qbar = np.kron(np.eye(N), self.Q)
        # ds28: 无约束二次规划闭式解 U* = -H⁻¹ f = -H⁻¹ G x0
        self.H = 2.0 * (L.T @ Qbar @ L + self.R * np.eye(m * N))
        self.H_inv = np.linalg.inv(self.H)
        self.G = 2.0 * L.T @ Qbar @ M

    def control(self, x):
        """输入当前状态 x=[theta, omega]（弧度），返回控制力矩 u（N·m，已限幅）。"""
        x = np.asarray(x, dtype=float).reshape(-1)
        U = -self.H_inv @ (self.G @ x)      # 未来 N 步最优控制序列
        return float(np.clip(U[0], -self.umax, self.umax))  # 取第一步 + 饱和限幅

    def reset(self):
        pass
