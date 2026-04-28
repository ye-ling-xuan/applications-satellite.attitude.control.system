import matplotlib.pyplot as plt
import numpy as np

# 1. 基本线图
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
plt.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦和余弦')
plt.legend()
plt.grid(True)
plt.savefig('sin_cos.png', dpi=150)
plt.show()

# 2. 散点图
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
colors = np.random.rand(50)
sizes = np.random.rand(50) * 100

plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.5)
plt.colorbar()
plt.title('散点图')
plt.show()

# 3. 柱状图
categories = ['PID', 'RL', 'LQR']
values = [2.5, 1.8, 2.2]
plt.bar(categories, values, color=['blue', 'red', 'green'])
plt.ylabel('稳定时间 (s)')
plt.title('控制器性能对比')
plt.show()

# 4. 子图布局
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].plot(x, y1)
axes[0,0].set_title('sin')
axes[0,1].plot(x, y2)
axes[0,1].set_title('cos')
axes[1,0].hist(np.random.randn(1000), bins=30)
axes[1,0].set_title('直方图')
axes[1,1].scatter(np.random.randn(100), np.random.randn(100))
axes[1,1].set_title('散点')
plt.tight_layout()
plt.savefig('subplots.png')
plt.show()