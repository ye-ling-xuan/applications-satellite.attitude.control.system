import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     #导入3D绘图工具
import matplotlib.animation as animation    #导入动画模块

fig = plt.figure()                          #创建一个新的图形窗口
ax = fig.add_subplot(111, projection='3d')  #添加为3D子图
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)#设置坐标轴范围
line, = ax.plot([],[],[], 'r-o') #坐标初始化，r-o：red，—实线，o圆圈标记

#定义函数，调用每一帧作为参数
def update(frame):
    angle = np.radians(frame)   #帧序号转换为角度
    x = [0, np.cos(angle)]      #计算向量终点坐标
    y = [0, np.sin(angle)]
    z = [0, 0]
    line.set_data(x, y)         #更新xy坐标
    line.set_3d_properties(z)   #更新z坐标
    return line,

ani = animation.FuncAnimation(fig, update, frames=360, interval=50)
plt.show()  #生成动画，fig图形对象，update更新函数，frame帧数一圈360°，interval每帧间隔50ms