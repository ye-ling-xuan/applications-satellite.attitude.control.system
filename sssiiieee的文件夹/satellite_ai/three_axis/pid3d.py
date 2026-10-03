"""
三通道独立 PID 控制器（satellite_ai 复制版）

ds13: 从 satellite1PID/week5：三轴卫星姿态控制/pid3d.py 复制而来，控制逻辑未改动，
      仅精简注释。用于三轴 PID vs 强化学习 的公平对比。
      输入三维误差向量（弧度），输出三维力矩 (N·m)，支持积分限幅与输出限幅。
"""

import numpy as np


class PID3D:
    def __init__(self, Kp, Ki, Kd, dt, output_limit=None, integral_limit=None):
        self.Kp = self._to_3d_array(Kp)
        self.Ki = self._to_3d_array(Ki)
        self.Kd = self._to_3d_array(Kd)
        self.dt = dt
        self.output_limit = None if output_limit is None else self._to_3d_array(output_limit)
        self.integral_limit = None if integral_limit is None else self._to_3d_array(integral_limit)
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    @staticmethod
    def _to_3d_array(value):
        """把标量或长度为 3 的序列统一为 shape=(3,) 数组。"""
        if np.isscalar(value):
            return np.full(3, value, dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return np.full(3, arr.item())
        if arr.size == 3:
            return arr
        raise ValueError("Input must be scalar or sequence of length 3")

    def compute(self, error_vec):
        """位置式 PID，error_vec 为三维误差（弧度）。"""
        self.integral += error_vec * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error_vec - self.prev_error) / self.dt
        output = self.Kp * error_vec + self.Ki * self.integral + self.Kd * derivative
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        self.prev_error = error_vec
        return output

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0
