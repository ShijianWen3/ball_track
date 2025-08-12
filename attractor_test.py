import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设 positions.shape = (N, 4, 3) ， index 0 是固定点（基点），1/2/3 是三个球
# positions = ... (你的数据)

def vec_angles(vec):
    # vec: (...,3) 向量
    x, y, z = vec[...,0], vec[...,1], vec[...,2]
    phi = np.arctan2(y, x)                        # 方位角
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))  # 俯仰角（-pi/2..pi/2）
    return theta, phi

def central_diff(x, dt):
    # 中心差分，边界用前向/后向
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

# --- 计算连杆向量和角度 ---
# u1 = p1 - p0, u2 = p2 - p1, u3 = p3 - p2
u1 = positions[:,1,:] - positions[:,0,:]
u2 = positions[:,2,:] - positions[:,1,:]
u3 = positions[:,3,:] - positions[:,2,:]

theta1, phi1 = vec_angles(u1)
theta2, phi2 = vec_angles(u2)
theta3, phi3 = vec_angles(u3)

# --- 角速度（以 theta 为例） ---
dt = time[1] - time[0]  # 你的采样间隔
omega_theta1 = central_diff(theta1, dt)
omega_theta2 = central_diff(theta2, dt)
omega_theta3 = central_diff(theta3, dt)

# --- 绘图 1: (theta1, omega_theta1) ---
plt.figure()
plt.plot(theta1, omega_theta1, '.', markersize=1)
plt.xlabel('theta1 (rad)')
plt.ylabel('omega_theta1 (rad/s)')
plt.title('Phase plot: theta1 vs omega_theta1')
plt.show()

# --- 延迟嵌入 3D: s(t), s(t+tau), s(t+2tau) ---
s = theta1  # 也可以用 positions[:,1,0] 等
tau = int(0.02 / dt)  # 例如 20 ms 延迟（根据自相关选择），这里只是示例
m = 3
T = len(s) - (m-1)*tau
emb = np.zeros((T, m))
for i in range(m):
    emb[:, i] = s[i*tau : i*tau + T]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(emb[:,0], emb[:,1], emb[:,2], lw=0.5)
ax.set_xlabel('s(t)')
ax.set_ylabel(f's(t+{tau}dt)')
ax.set_zlabel(f's(t+{2*tau}dt)')
plt.title('Delay embedding (3D)')
plt.show()

# --- Poincaré: 记录 theta1, omega1 当 theta3 由负变为正（过零且上升） ---
cross_idx = np.where((theta3[:-1] < 0) & (theta3[1:] >= 0))[0]
# 可以插值更精确地找到穿过点，但简单取索引2即可
poincare_theta1 = theta1[cross_idx]
poincare_omega1 = omega_theta1[cross_idx]

plt.figure()
plt.plot(poincare_theta1, poincare_omega1, '.', markersize=2)
plt.xlabel('theta1 at section')
plt.ylabel('omega1 at section')
plt.title('Poincaré section (theta3=0, dtheta3/dt>0)')
plt.show()
