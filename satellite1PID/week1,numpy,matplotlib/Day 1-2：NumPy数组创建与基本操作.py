import numpy as np

# 1. 创建数组
a = np.array([1, 2, 3, 4, 5])           # 一维数组
b = np.array([[1, 2], [3, 4]])           # 二维数组
c = np.zeros((3, 3))                     # 全0矩阵
d = np.ones((2, 4))                      # 全1矩阵
e = np.eye(3)                             # 单位矩阵
f = np.linspace(0, 10, 5)                 # 从0到10等间隔取5个数
g = np.arange(0, 10, 2)                   # 从0开始步长为2到10

print("a:", a)
print("b.shape:", b.shape)
print("c:\n", c)

# 2. 索引和切片
print("a[2]:", a[2])                      # 第三个元素
print("a[1:4]:", a[1:4])                  # 切片
print("b[0,1]:", b[0, 1])                  # 第1行第2列

# 3. 数组运算
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("加法:", x + y)
print("点乘:", np.dot(x, y))               # 向量点积
print("叉乘:", np.cross(x, y))              # 向量叉积