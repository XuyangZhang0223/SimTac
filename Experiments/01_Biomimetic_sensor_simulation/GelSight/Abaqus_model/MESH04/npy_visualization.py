import numpy as np
import matplotlib.pyplot as plt

# 加载.npy数据
data = np.load('MESH04.npy')
print(data.shape)
print(data[:5])

# 提取坐标数据
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# 创建三维图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
ax.scatter3D(x, y, z, c=z, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
#plt.savefig("npy_visualization.png")
