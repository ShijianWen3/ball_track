import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch

class TriplePendulumSystem:
    def __init__(self):
        # 系统参数（从MATLAB代码转换）
        self.L1 = 0.3302
        self.L2 = 0.1591 
        self.L3 = 0.1815
        
        self.d1 = 0.8 * self.L1
        self.d1p = 0.2 * self.L1
        self.d2 = 0.682537465 * self.L2
        self.d2p = 0.317462535 * self.L2
        self.l3 = self.L3 / 2
        # self.l3 = self.L3 / 3
        
        rho = 0.115394883
        self.mr1 = rho * self.L1
        self.mr2 = rho * self.L2  
        self.mr3 = rho * self.L3
        
        self.mB = 0.0406369255
        self.mC = 0.0406369255
        self.m = 0.2  # 可调整的质量参数
        
        # 增加阻尼以获得更清晰的吸引子
       
        self.b1 = 3e-3  # 增加阻尼
        self.b2 = 3e-3
        self.b3 = 3e-3
        
        self.Ro = 3e-3
        self.Ri = 2e-3
        self.g = 9.81
        
        # 惯性矩
        self.Ipar1 = (1/8) * self.mr1 * (self.Ro**2 + self.Ri**2) + self.m * self.d1**2
        self.Iperp1 = (1/4) * self.mr1 * (self.Ro**2 + self.Ri**2) + (1/12) * self.mr1 * self.L1**2 + self.m * self.d1**2
        self.Ipar2 = (1/8) * self.mr2 * (self.Ro**2 + self.Ri**2)
        self.Iperp2 = (1/4) * self.mr2 * (self.Ro**2 + self.Ri**2) + (1/12) * self.mr2 * self.L2**2
        self.Ipar3 = (1/8) * self.mr3 * (self.Ro**2 + self.Ri**2)
        self.Iperp3 = (1/4) * self.mr3 * (self.Ro**2 + self.Ri**2) + (1/12) * self.mr3 * self.L3**2
    
    def wrap_angle(self, angle):
        """将角度包装到 [-π, π] 范围"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def equations_of_motion(self, t, state):
        """改进的运动方程，使用更强的非线性耦合"""
        q = state[:3]  # [theta1, phi2, phi3]
        dq = state[3:]  # [dtheta1, dphi2, dphi3]
        
        theta1, phi2, phi3 = q
        dtheta1, dphi2, dphi3 = dq
        
        # 增强非线性耦合的运动方程
        g_eff = self.g / self.L1
        
        # 更复杂的耦合项
        coupling12 = 0.5 * np.sin(2*(phi2 - theta1)) * dphi2**2
        coupling23 = 0.3 * np.sin(phi3 - phi2) * dphi3**2
        coupling13 = 0.2 * np.sin(phi3 - theta1) * dphi3 * dtheta1
        
        # θ1方程：主摆动
        ddtheta1 = (-g_eff * np.sin(theta1) 
                    - coupling12 - coupling13
                    - self.b1 * dtheta1)
        
        # φ2方程：二级摆动，与θ1强耦合
        ddphi2 = (-g_eff * 0.8 * np.sin(phi2) * np.cos(theta1)
                  + 0.4 * np.sin(theta1 - phi2) * dtheta1**2
                  - coupling23
                  - self.b2 * dphi2)
        
        # φ3方程：三级摆动，与前两者耦合
        ddphi3 = (-g_eff * 0.6 * np.sin(phi3)
                  + 0.2 * np.sin(phi2 - phi3) * dphi2**2
                  + 0.1 * np.sin(theta1 - phi3) * dtheta1**2
                  - self.b3 * dphi3)
        
        return np.array([dtheta1, dphi2, dphi3, ddtheta1, ddphi2, ddphi3])
    
    def simulate(self, initial_conditions, t_span, dt=0.01):
        """高精度仿真"""
        t_eval = np.arange(0, t_span, dt)
        sol = solve_ivp(self.equations_of_motion, [0, t_span], initial_conditions, 
                       t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
        return sol.t, sol.y.T
    
    def to_cartesian_coords(self, traj):
        """直接返回角度和角速度，不做正余弦变换"""
        theta1, phi2, phi3 = traj[:, 0], traj[:, 1], traj[:, 2]
        dtheta1, dphi2, dphi3 = traj[:, 3], traj[:, 4], traj[:, 5]
        return np.column_stack([theta1, phi2, phi3, dtheta1, dphi2, dphi3])
    
    def find_poincare_section(self, trajectory, section_var=0, section_value=0, direction='positive'):
        """改进的Poincaré截面"""
        crossings = []
        for i in range(1, len(trajectory)):
            if direction == 'positive':
                if (trajectory[i-1, section_var] < section_value and 
                    trajectory[i, section_var] >= section_value and
                    trajectory[i, section_var + 3] > 0):  # 正向穿越
                    crossings.append(i)
            else:
                if (trajectory[i-1, section_var] > section_value and 
                    trajectory[i, section_var] <= section_value and
                    trajectory[i, section_var + 3] < 0):  # 负向穿越
                    crossings.append(i)
        return crossings

def plot_enhanced_attractors():
    """绘制增强的吸引子图像"""
    system = TriplePendulumSystem()
    initial_conditions_list = [
       
        [0.8, -0.4, 1.8, 0, 0, 0]
        
    ]
    colors = ['red', 'blue', 'green', 'purple']
    
    # 更长的仿真时间和更高精度
    sim_time = 200  # 增加仿真时间
    transient_time = 120  # 丢弃瞬态
    
    # fig1 = plt.figure(figsize=(16, 12))
    
    # # 1. θ₁ 相空间轨迹（直接用角度）
    # plt.subplot(2, 4, 1)
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     plt.plot(traj[start_idx:, 0], traj[start_idx:, 1], color=colors[i], 
    #             alpha=0.7, linewidth=0.5, markersize=0.1)
    # plt.xlabel('θ₁ (rad)')
    # plt.ylabel('φ₂ (rad)')
    # plt.title('θ₁-φ₂')
    # plt.grid(True, alpha=0.3)
    
    # # 2. θ₁ vs dθ₁/dt
    # plt.subplot(2, 4, 2)
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     plt.plot(traj[start_idx:, 0], traj[start_idx:, 3], 
    #             color=colors[i], alpha=0.7, linewidth=0.5)
    # plt.xlabel('θ₁ (rad)')
    # plt.ylabel('dθ₁/dt (rad/s)')
    # plt.title('θ₁-dθ₁/dt')
    # plt.grid(True, alpha=0.3)
    
    # # 3. φ₂ vs φ₃
    # plt.subplot(2, 4, 3)
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     plt.plot(traj[start_idx:, 1], traj[start_idx:, 3], color=colors[i], 
    #             alpha=0.7, linewidth=0.5)
    # plt.xlabel('φ₂ (rad)')
    # plt.ylabel('dθ₁/dt (rad/s)')
    # plt.title('φ₂-dθ₁/dt')
    # plt.grid(True, alpha=0.3)

    # ax = fig1.add_subplot(2, 4, 4, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):  
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 0], traj[start_idx:, 1], traj[start_idx:, 3], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('θ₁ (rad)')
    # ax.set_ylabel('φ₂ (rad)')
    # ax.set_zlabel('dθ₁/dt (rad/s)')
    # ax.set_title('3D-θ₁-φ₂-dθ₁/dt')

    # # 5. 高质量Poincaré截面（直接用角度）
    # plt.subplot(2, 4, 5)
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, 300, dt=0.01)
    #     crossings = system.find_poincare_section(traj, section_var=0, section_value=0, direction='positive')
    #     if len(crossings) > 20:
    #         poincare_points = traj[crossings[10:], :]
    #         plt.scatter(poincare_points[:, 1], poincare_points[:, 4], color=colors[i], alpha=0.8, s=1)
    # plt.xlabel('φ₂ (rad)')
    # plt.ylabel('dφ₂/dt (rad/s)')
    # plt.title('Poincaré截面 (θ₁=0, 上升)')
    # plt.grid(True, alpha=0.3)
    
    # # 6. 多变量时间序列
    # plt.subplot(2, 4, 6)
    # initial_cond = [1.2, -0.8, 0.6, -0.2, 0.3, -0.1]
    # t, traj = system.simulate(initial_cond, 100, dt=0.02)
    # plt.plot(t[1000:3000], traj[1000:3000, 0], 'r-', label='θ₁', alpha=0.8, linewidth=0.8)
    # plt.plot(t[1000:3000], traj[1000:3000, 1], 'b-', label='φ₂', alpha=0.8, linewidth=0.8)
    # plt.plot(t[1000:3000], traj[1000:3000, 2], 'g-', label='φ₃', alpha=0.8, linewidth=0.8)
    # plt.xlabel('时间 (s)')
    # plt.ylabel('角度 (rad)')
    # plt.title('长期时间序列')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # # 7. 增强功率谱
    # plt.subplot(2, 4, 7)
    # initial_cond = [1.2, -0.8, 0.6, -0.2, 0.3, -0.1]
    # t, traj = system.simulate(initial_cond, 200, dt=0.01)
    # # 使用更长的数据进行谱分析
    # f, psd = welch(traj[5000:, 0], fs=100, nperseg=2048, noverlap=1024)
    # plt.loglog(f[1:], psd[1:], 'r-', alpha=0.8)
    # plt.xlabel('频率 (Hz)')
    # plt.ylabel('功率谱密度')
    # plt.title('θ₁的功率谱 (对数坐标)')
    # plt.grid(True, alpha=0.3)
    
    # # 8. 轨道分离（Lyapunov特性）
    # plt.subplot(2, 4, 8)
    # eps = 1e-8
    # initial1 = [1.2, -0.8, 0.6, -0.2, 0.3, -0.1]
    # initial2 = [1.2 + eps, -0.8, 0.6, -0.2, 0.3, -0.1]
    
    # t, traj1 = system.simulate(initial1, 50, dt=0.01)
    # t, traj2 = system.simulate(initial2, 50, dt=0.01)
    
    # separation = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
    # # 避免对0取对数
    # separation = np.maximum(separation, 1e-15)
    
    # plt.semilogy(t[100:2000], separation[100:2000], 'k-', linewidth=1)
    # plt.xlabel('时间 (s)')
    # plt.ylabel('轨迹分离距离')
    # plt.title('混沌敏感性')
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()


    # fig2 = plt.figure(figsize=(16, 12))
    # #3D投影 - 角度空间
    # ax = fig2.add_subplot(2, 4, 1, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 0], traj[start_idx:, 1], traj[start_idx:, 2], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('θ₁ (rad)')
    # ax.set_ylabel('φ₂ (rad)')
    # ax.set_zlabel('φ₃ (rad)')
    # ax.set_title('3D-θ₁-φ₂-φ₃')

    # ax = fig2.add_subplot(2, 4, 2, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 3], traj[start_idx:, 4], traj[start_idx:, 5], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('dθ₁/dt (rad/s)')
    # ax.set_ylabel('dφ₂/dt (rad/s)')
    # ax.set_zlabel('dφ₃/dt (rad/s)')
    # ax.set_title('3D-w1-w2-w3')

    # ax = fig2.add_subplot(2, 4, 3, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):  
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 0], traj[start_idx:, 1], traj[start_idx:, 3], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('θ₁ (rad)')
    # ax.set_ylabel('φ₂ (rad)')
    # ax.set_zlabel('dθ₁/dt (rad/s)')
    # ax.set_title('3D-θ₁-φ₂-dθ₁/dt')

    # ax = fig2.add_subplot(2, 4, 4, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 2], traj[start_idx:, 4], traj[start_idx:, 5], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('φ₃ (rad)')
    # ax.set_ylabel('dφ₂/dt (rad/s)')
    # ax.set_zlabel('dφ₃/dt (rad/s)')
    # ax.set_title('3D-φ₃-dφ₂/dt-dφ₃/dt')

    # ax = fig2.add_subplot(2, 4, 5, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 1], traj[start_idx:, 2], traj[start_idx:, 4], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('φ₂ (rad)')
    # ax.set_ylabel('φ₃ (rad)')
    # ax.set_zlabel('dφ₂/dt (rad/s)')
    # ax.set_title('3D-φ₂-φ₃-dφ₂/dt')

    # ax = fig2.add_subplot(2, 4, 6, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 0], traj[start_idx:, 3], traj[start_idx:, 5], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('θ₁ (rad)')
    # ax.set_ylabel('dθ₁/dt (rad/s)')
    # ax.set_zlabel('dφ₃/dt (rad/s)')
    # ax.set_title('3D-θ₁-dθ₁/dt-dφ₃/dt')

    # ax = fig2.add_subplot(2, 4, 7, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 1], traj[start_idx:, 3], traj[start_idx:, 5], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('φ₂ (rad)')
    # ax.set_ylabel('dφ₂/dt (rad/s)')
    # ax.set_zlabel('dφ₃/dt (rad/s)')
    # ax.set_title('3D-φ₂-dφ₂/dt-dφ₃/dt')

    # ax = fig2.add_subplot(2, 4, 8, projection='3d')
    # for i, initial_cond in enumerate(initial_conditions_list):
    #     t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
    #     start_idx = int(transient_time / 0.02)
    #     ax.plot(traj[start_idx:, 0], traj[start_idx:, 2], traj[start_idx:, 4], color=colors[i], alpha=0.6, linewidth=0.5)
    # ax.set_xlabel('θ₁ (rad)')
    # ax.set_ylabel('φ₃ (rad)')
    # ax.set_zlabel('dφ₂/dt (rad/s)')
    # ax.set_title('3D-θ₁-φ₃-dφ₂/dt')


    # plt.tight_layout()


    # 自动生成20种三维投影，分布在5个fig，每个fig4个子图
    var_names = [r'$\theta_1$', r'$\varphi_2$', r'$\varphi_3$', r'$d\theta_1/dt$', r'$d\varphi_2/dt$', r'$d\varphi_3/dt$']
    var_labels = ['θ₁ (rad)', 'φ₂ (rad)', 'φ₃ (rad)', 'dθ₁/dt (rad/s)', 'dφ₂/dt (rad/s)', 'dφ₃/dt (rad/s)']
    n_vars = 6
    from itertools import combinations
    combs = list(combinations(range(n_vars), 3))  # 20种三元组
    fig_list = []
    for fig_idx in range(5):
        fig = plt.figure(figsize=(14, 10))
        fig_list.append(fig)
        for sub_idx in range(4):
            comb_idx = fig_idx * 4 + sub_idx
            if comb_idx >= len(combs):
                break
            i, j, k = combs[comb_idx]
            ax = fig.add_subplot(2, 2, sub_idx+1, projection='3d')
            for ini_i, initial_cond in enumerate(initial_conditions_list):
                t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
                start_idx = int(transient_time / 0.02)
                ax.plot(traj[start_idx:, i], traj[start_idx:, j], traj[start_idx:, k], color=colors[ini_i], alpha=0.7, linewidth=0.5)
            ax.set_xlabel(var_labels[i])
            ax.set_ylabel(var_labels[j])
            # 只在3D坐标轴对象上调用set_zlabel，防止2D子图报错
            from mpl_toolkits.mplot3d import Axes3D
            if isinstance(ax, Axes3D):
                ax.set_zlabel(var_labels[k])
            ax.set_title(f'3D-{var_labels[i]}-{var_labels[j]}-{var_labels[k]}')
        fig.suptitle(f'fig2_{fig_idx+1}: 6D-3D({fig_idx*4+1}-{min((fig_idx+1)*4,20)})', fontsize=16)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    fig3 = plt.figure(figsize=(16, 12))
    plt.subplot(1, 3, 1)
    for i, initial_cond in enumerate(initial_conditions_list):
        t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
        start_idx = int(transient_time / 0.02)
        plt.plot(traj[start_idx:, 0], traj[start_idx:, 3], color=colors[i], 
                alpha=0.7, linewidth=0.5, markersize=0.1)
    plt.xlabel('θ₁ (rad)')
    plt.ylabel('dθ₁/dt (rad)')
    plt.title('θ₁-dθ₁/dt')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    for i, initial_cond in enumerate(initial_conditions_list):
        t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
        start_idx = int(transient_time / 0.02)
        plt.plot(traj[start_idx:, 1], traj[start_idx:, 4], 
                color=colors[i], alpha=0.7, linewidth=0.5)
    plt.xlabel('φ₂ (rad)')
    plt.ylabel('dφ₂/dt (rad/s)')
    plt.title('φ₂-dφ₂/dt')
    plt.grid(True, alpha=0.3)
    
    # 3. φ₂ vs φ₃
    plt.subplot(1, 3, 3)
    for i, initial_cond in enumerate(initial_conditions_list):
        t, traj = system.simulate(initial_cond, sim_time, dt=0.02)
        start_idx = int(transient_time / 0.02)
        plt.plot(traj[start_idx:, 2], traj[start_idx:, 5], color=colors[i], 
                alpha=0.7, linewidth=0.5)
    plt.xlabel('φ₃ (rad)')
    plt.ylabel('dφ₃/dt (rad/s)')
    plt.title('φ₃-dφ₃/dt')
    plt.grid(True, alpha=0.3)
    
    plt.show()

def plot_parameter_scan():
    """参数扫描寻找最佳混沌区域"""
    system = TriplePendulumSystem()
    
    # 扫描不同的质量参数
    m_values = np.linspace(0.05, 1.0, 10)
    max_lyapunov = []
    
    print("正在扫描参数以寻找强混沌区域...")
    
    for m in m_values:
        system.m = m
        # 固定初始条件
        initial_cond = [1.0, 0.5, -0.3, 0.1, -0.2, 0.15]
        
        # 计算简单的轨道发散估计
        eps = 1e-8
        initial1 = initial_cond
        initial2 = [initial_cond[0] + eps] + initial_cond[1:]
        
        t, traj1 = system.simulate(initial1, 30, dt=0.02)
        t, traj2 = system.simulate(initial2, 30, dt=0.02)
        
        separation = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
        
        # 估算Lyapunov指数
        valid_idx = (separation > 1e-12) & (separation < 1e-2)
        if np.sum(valid_idx) > 10:
            log_sep = np.log(separation[valid_idx] / eps)
            lyap_est = np.mean(np.diff(log_sep) / 0.02)
        else:
            lyap_est = 0
            
        max_lyapunov.append(lyap_est)
        print(f"m = {m:.3f}, 估计λ_max = {lyap_est:.4f}")
    
    # 绘制参数扫描结果
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, max_lyapunov, 'bo-', markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='λ=0 (周期边界)')
    plt.xlabel('质量参数 m')
    plt.ylabel('估计的最大Lyapunov指数')
    plt.title('参数扫描：寻找混沌区域')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 标记最佳参数
    best_idx = np.argmax(max_lyapunov)
    plt.annotate(f'最佳: m={m_values[best_idx]:.3f}', 
                xy=(m_values[best_idx], max_lyapunov[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()
    
    return m_values[best_idx]

if __name__ == "__main__":
    print("=== 三自由度摆球系统混沌吸引子分析 ===")
    
    # 1. 首先找到最佳参数
    # print("\n1. 参数扫描...")
    # best_m = plot_parameter_scan()
    best_m = 0.3
    
    # 2. 使用最佳参数绘制吸引子
    print(f"\n2. 使用最佳参数 m={best_m:.3f} 绘制吸引子...")
    system = TriplePendulumSystem()
    system.m = best_m
    plot_enhanced_attractors()
    
    print("\n=== 图像解读 ===")
    print("1. 前三图：使用cos/sin变换避免角度包装问题")
    print("2. 3D图：真正的吸引子几何结构")  
    print("3. Poincaré截面：吸引子的精细分形结构")
    print("4. 时间序列：非周期混沌行为")
    print("5. 功率谱：连续谱表明混沌特性")
    print("6. 轨道分离：指数发散证明混沌敏感性")
    print("\n如果吸引子仍不明显，尝试：")
    print("- 调整initial_conditions_list中的初值")  
    print("- 增加sim_time（仿真时间）")
    print("- 减小阻尼系数b1,b2,b3")