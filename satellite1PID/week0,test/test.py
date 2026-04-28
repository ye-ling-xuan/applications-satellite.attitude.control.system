import numpy as np
import matplotlib.pyplot as plt

print("NumPy版本:", np.__version__)
print("Matplotlib版本:", plt.matplotlib.__version__)

# 测试绘图
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x))
plt.title("测试成功！")
plt.show()