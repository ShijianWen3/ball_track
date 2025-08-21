import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ChaoticAttractors:
    def __init__(self):
        self.dt = 0.01
        self.t_span = (0, 50)
        self.t_eval = np.arange(0, 50, self.dt)
    
    def lorenz_system(self, t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        """洛伦兹方程组 - 单吸引子"""
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
    def rossler_system(self, t, state, a=0.2, b=0.2, c=5.7):
        """Rössler方程组 - 单吸引子"""
        x, y, z = state
        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
        return [dxdt, dydt, dzdt]
    
    def chua_system(self, t, state, alpha=10.0, beta=14.87, m0=-1.27, m1=-0.68):
        """蔡氏电路方程 - 双涡卷吸引子"""
        x, y, z = state
        # 分段线性函数
        f_x = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
        dxdt = alpha * (y - x - f_x)
        dydt = x - y + z
        dzdt = -beta * y
        return [dxdt, dydt, dzdt]
    
    def double_scroll_system(self, t, state, a=0.7, b=0.1):
        """双涡卷系统"""
        x, y, z = state
        dxdt = y
        dydt = z
        dzdt = -a * z - y + np.tanh(x) - b * np.tanh(x)**3
        return [dxdt, dydt, dzdt]
    
    def duffing_system(self, t, state, alpha=-1, beta=1, delta=0.1, gamma=0.3, omega=1.2):
        """杜芬振子 - 可产生混沌"""
        x, y = state
        dxdt = y
        dydt = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
        return [dxdt, dydt]
    
    def thomas_system(self, t, state, b=0.19):
        """Thomas方程组 - 单吸引子"""
        x, y, z = state
        dxdt = np.sin(y) - b * x
        dydt = np.sin(z) - b * y
        dzdt = np.sin(x) - b * z
        return [dxdt, dydt, dzdt]
    
    def solve_system(self, system_func, initial_condition, **kwargs):
        """求解动力学系统"""
        sol = solve_ivp(system_func, self.t_span, initial_condition, 
                       t_eval=self.t_eval, args=tuple(kwargs.values()),
                       method='RK45', rtol=1e-8, atol=1e-10)
        return sol.y
    
    def plot_3d_attractor(self, trajectory, title, color='blue', alpha=0.7):
        """绘制3D吸引子"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x, y, z = trajectory[:3]  # 只取前3个维度
        ax.plot(x, y, z, color=color, alpha=alpha, linewidth=0.5)
        ax.scatter(x[0], y[0], z[0], color='red', s=50, label='起始点')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        return fig, ax
    
    def plot_2d_attractor(self, trajectory, title, color='blue', alpha=0.7):
        """绘制2D吸引子"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x, y = trajectory[:2]
        ax.plot(x, y, color=color, alpha=alpha, linewidth=0.5)
        ax.scatter(x[0], y[0], color='red', s=50, label='起始点')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        return fig, ax
    
    def plot_phase_portrait(self, trajectory, title, dims=[0, 1]):
        """绘制相图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = trajectory[dims[0]]
        y = trajectory[dims[1]]
        
        # 使用颜色渐变显示时间演化
        colors = np.linspace(0, 1, len(x))
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=0.5, alpha=0.6)
        
        ax.set_xlabel(f'X{dims[0]+1}')
        ax.set_ylabel(f'X{dims[1]+1}')
        ax.set_title(title)
        plt.colorbar(scatter, label='时间')
        return fig, ax

def generate_all_attractors():
    """生成所有吸引子图像"""
    ca = ChaoticAttractors()
    
    # 1. 洛伦兹吸引子 - 经典单吸引子
    print("生成洛伦兹吸引子...")
    lorenz_traj = ca.solve_system(ca.lorenz_system, [1, 1, 1])
    fig1, _ = ca.plot_3d_attractor(lorenz_traj, '洛伦兹吸引子 (单吸引子)', 'blue')
    
    # 2. Rössler吸引子 - 单吸引子
    print("生成Rössler吸引子...")
    rossler_traj = ca.solve_system(ca.rossler_system, [1, 1, 1])
    fig2, _ = ca.plot_3d_attractor(rossler_traj, 'Rössler吸引子 (单吸引子)', 'green')
    
    # 3. 蔡氏电路 - 双涡卷吸引子
    print("生成蔡氏电路双涡卷吸引子...")
    chua_traj = ca.solve_system(ca.chua_system, [0.1, 0.1, 0.1])
    fig3, _ = ca.plot_3d_attractor(chua_traj, '蔡氏电路 (双涡卷吸引子)', 'red')
    
    # 4. Thomas吸引子 - 单吸引子
    print("生成Thomas吸引子...")
    thomas_traj = ca.solve_system(ca.thomas_system, [0.1, 0.1, 0.1])
    fig4, _ = ca.plot_3d_attractor(thomas_traj, 'Thomas吸引子 (单吸引子)', 'purple')
    
    # 5. 杜芬振子 - 2D混沌
    print("生成杜芬振子...")
    ca.t_span = (0, 100)  # 杜芬振子需要更长时间
    ca.t_eval = np.arange(0, 100, ca.dt)
    duffing_traj = ca.solve_system(ca.duffing_system, [1, 0])
    fig5, _ = ca.plot_2d_attractor(duffing_traj, '杜芬振子 (混沌)', 'orange')
    
    # 6. 相图对比
    print("生成相图对比...")
    ca.t_span = (0, 50)
    ca.t_eval = np.arange(0, 50, ca.dt)
    
    fig6, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 洛伦兹相图 (x-y)
    lorenz_traj = ca.solve_system(ca.lorenz_system, [1, 1, 1])
    axes[0,0].plot(lorenz_traj[0], lorenz_traj[1], 'b-', alpha=0.7, linewidth=0.5)
    axes[0,0].set_title('洛伦兹 X-Y 相图')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    
    # 洛伦兹相图 (x-z)
    axes[0,1].plot(lorenz_traj[0], lorenz_traj[2], 'b-', alpha=0.7, linewidth=0.5)
    axes[0,1].set_title('洛伦兹 X-Z 相图')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Z')
    
    # 蔡氏电路相图 (x-y)
    chua_traj = ca.solve_system(ca.chua_system, [0.1, 0.1, 0.1])
    axes[1,0].plot(chua_traj[0], chua_traj[1], 'r-', alpha=0.7, linewidth=0.5)
    axes[1,0].set_title('蔡氏电路 X-Y 相图 (双涡卷)')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    
    # Rössler相图 (x-y)
    rossler_traj = ca.solve_system(ca.rossler_system, [1, 1, 1])
    axes[1,1].plot(rossler_traj[0], rossler_traj[1], 'g-', alpha=0.7, linewidth=0.5)
    axes[1,1].set_title('Rössler X-Y 相图')
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('Y')
    
    plt.tight_layout()
    
    # 7. 时间序列对比
    print("生成时间序列对比...")
    fig7, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    t = ca.t_eval[:2000]  # 只显示前2000个点
    
    # 洛伦兹时间序列
    axes[0].plot(t, lorenz_traj[0][:2000], 'b-', linewidth=0.8)
    axes[0].set_title('洛伦兹系统 - X分量时间序列')
    axes[0].set_ylabel('X')
    axes[0].grid(True, alpha=0.3)
    
    # 蔡氏电路时间序列
    axes[1].plot(t, chua_traj[0][:2000], 'r-', linewidth=0.8)
    axes[1].set_title('蔡氏电路 - X分量时间序列')
    axes[1].set_ylabel('X')
    axes[1].grid(True, alpha=0.3)
    
    # Rössler时间序列
    axes[2].plot(t, rossler_traj[0][:2000], 'g-', linewidth=0.8)
    axes[2].set_title('Rössler系统 - X分量时间序列')
    axes[2].set_xlabel('时间')
    axes[2].set_ylabel('X')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7]

if __name__ == "__main__":
    # 生成所有吸引子图像
    figures = generate_all_attractors()
    
    # 显示所有图像
    plt.show()
    
    print("\n混沌吸引子特征说明:")
    print("1. 洛伦兹吸引子: 经典的单吸引子,呈蝴蝶形状")
    print("2. Rössler吸引子: 螺旋形单吸引子")  
    print("3. 蔡氏电路: 双涡卷吸引子,具有两个涡卷结构")
    print("4. Thomas吸引子: 具有周期性的单吸引子")
    print("5. 杜芬振子: 2D混沌系统")
    print("6. 相图展示了系统在不同维度上的投影")
    print("7. 时间序列展示了混沌系统的非周期性特征")