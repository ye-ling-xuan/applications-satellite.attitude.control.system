"""
三通道独立 PID 控制器，输入为三维误差向量（弧度），输出为三维力矩 (N·m)。
支持积分限幅、输出限幅。
"""
import numpy as np

class PID3D:
    def __init__(self, Kp, Ki, Kd, dt,
                 output_limit=None, integral_limit=None):
        """
        参数:
            Kp, Ki, Kd: 可以是标量（三轴相同）或长度为 3 的序列
            dt: 采样周期 (s)
            output_limit: 每个通道输出绝对值上限 (N·m)，标量或序列
            integral_limit: 每个通道积分累加器上限 (rad·s)，标量或序列
        """
        '''
        让 PID 控制器能同时处理 x/y/z 三个轴的控制，
        兼容 “三轴共用一组参数” 和 “三轴独立调参” 两种场景。
        _to_3d_array 这个函数的作用，是把用户输入的参数，
        不管是标量还是序列，都转换成 shape=(3,) 的 NumPy 数组：
        如果用户传标量，比如 Kp=2.0 → 自动变成 np.array([2.0, 2.0, 2.0])，三个轴共用同一个比例系数；
        如果用户传序列，比如 Kp=[2.0, 1.5, 1.0] → 直接变成 np.array([2.0, 1.5, 1.0])，三个轴用不同的比例系数。
        '''
        # 将 Kp, Ki, Kd 统一为长度为 3 的 numpy 数组
        self.Kp = self._to_3d_array(Kp)
        self.Ki = self._to_3d_array(Ki)
        self.Kd = self._to_3d_array(Kd)

        self.dt = dt

        # 处理输出限幅
        if output_limit is None:
            self.output_limit = None
        else:
            self.output_limit = self._to_3d_array(output_limit)

        # 处理积分限幅
        if integral_limit is None:
            self.integral_limit = None
        else:
            self.integral_limit = self._to_3d_array(integral_limit)

        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
    """
    @staticmethod 是 Python 中的静态方法装饰器
    静态方法就是 “挂在类里面的普通函数”，只是为了代码归类更清晰，把和类相关的工具函数放在类内部，而不是散落在全局作用域
    """
    @staticmethod
    def _to_3d_array(value):
        """将输入转换为长度为 3 的 numpy 数组。"""
        if np.isscalar(value):
            return np.full(3, value, dtype=float)
        else:
            arr = np.asarray(value, dtype=float)
            if arr.size == 1:
                return np.full(3, arr.item())
            elif arr.size == 3:
                return arr
            else:
                raise ValueError("Input must be scalar or sequence of length 3")

    def compute(self, error_vec):
        """
        输入: error_vec 三维误差向量 (弧度)
        输出: 三维力矩 (N·m)
        """
        # 积分
        self.integral += error_vec * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # 微分
        derivative = (error_vec - self.prev_error) / self.dt

        # PID 输出
        output = self.Kp * error_vec + self.Ki * self.integral + self.Kd * derivative

        # 输出限幅
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error_vec
        return output

    def reset(self):
        """重置积分和上一时刻误差"""
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0