import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
line, = ax.plot([],[],[], 'r-o')

def update(frame):
    angle = np.radians(frame)
    x = [0, np.cos(angle)]
    y = [0, np.sin(angle)]
    z = [0, 0]
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

ani = animation.FuncAnimation(fig, update, frames=360, interval=50)
plt.show()