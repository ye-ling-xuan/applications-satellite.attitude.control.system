import numpy as np
# 矩阵运算
A = np.array([[3, 1], [1, 2]])
B = np.array([[2, 0], [1, 3]])

print("A @ B:\n", A @ B)                  # 矩阵乘法
print("A.T:\n", A.T)                       # 转置
print("A的逆:\n", np.linalg.inv(A))         # 逆矩阵

# 解线性方程组 Ax = b
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print("解 x =", x)

# 统计
data = np.random.randn(1000)               # 1000个正态分布随机数
print("均值:", np.mean(data))
print("标准差:", np.std(data))
print("最大值:", np.max(data))
print("最小值:", np.min(data))