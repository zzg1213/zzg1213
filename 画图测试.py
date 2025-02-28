import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# 创建一个包含三个子图的布局
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 第一个子图：正弦曲线
axes[0].plot(x, y1, label='sin(x)', color='blue')
axes[0].set_title('Sine Curve')
axes[0].set_xlabel('X')
axes[0].set_ylabel('sin(x)')
axes[0].legend()
axes[0].grid(True)

# 第二个子图：余弦曲线
axes[1].plot(x, y2, label='cos(x)', color='green')
axes[1].set_title('Cosine Curve')
axes[1].set_xlabel('X')
axes[1].set_ylabel('cos(x)')
axes[1].legend()
axes[1].grid(True)

# 第三个子图：正切曲线
axes[2].plot(x, y3, label='tan(x)', color='red', linestyle='--')
axes[2].set_title('Tangent Curve')
axes[2].set_xlabel('X')
axes[2].set_ylabel('tan(x)')
axes[2].legend()
axes[2].grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()