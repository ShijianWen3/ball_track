import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

class LagrangianChaoticSystem:
    """基于拉格朗日方程的严谨三自由度混沌系统"""
    
    def __init__(self):
        """初始化系统参数，基于MATLAB代码中的物理参数"""
        # 几何参数
        self.L1 = 0.3302  # 第一段长度
        self.L2 = 0.1591  # 第二段长度  
        self.L3 = 0.1815  # 第三段长度
        
        # 质心位置参数
        self.d1 = 0.8 * self.L1
        self.d1p = 0.2 * self.L1
        self.d2 = 0.684491397 * self.L2
        self.d2p = 0.315508633 * self.L2
        self.l3 = self.L3 / 2
        
        # 质量参数
        rho = 0.115394883  # 杆的线密度
        self.mr1 = rho * self.L1  # 杆1质量
        self.mr2 = rho * self.L2  # 杆2质量
        self.mr3 = rho * self.L3  # 杆3质量
        self.mB = 0.0406369255    # 连接质量B
        self.mC = 0.0406369255    # 连接质量C
        self.m = 0.2              # 集中质量
        
        # 阻尼系数
        self.b1 = 1e-3
        self.b2 = 1e-3  
        self.b3 = 1e-3
        
        # 几何参数
        self.Ro = 3e-3  # 外半径
        self.Ri = 2e-3  # 内半径
        self.g = 9.81   # 重力加速度
        
        # 转动惯量
        self._compute_inertias()
        
        # 预计算重力系数
        self._compute_gravity_coefficients()
        
    def _compute_inertias(self):
        """计算转动惯量"""
        # 平行轴转动惯量
        self.Ipar1 = (1/8) * self.mr1 * (self.Ro**2 + self.Ri**2) + self.m * self.d1**2
        self.Ipar2 = (1/8) * self.mr2 * (self.Ro**2 + self.Ri**2)
        self.Ipar3 = (1/8) * self.mr3 * (self.Ro**2 + self.Ri**2)
        
        # 垂直轴转动惯量
        self.Iperp1 = (1/4) * self.mr1 * (self.Ro**2 + self.Ri**2) + (1/12) * self.mr1 * self.L1**2 + self.m * self.d1**2
        self.Iperp2 = (1/4) * self.mr2 * (self.Ro**2 + self.Ri**2) + (1/12) * self.mr2 * self.L2**2
        self.Iperp3 = (1/4) * self.mr3 * (self.Ro**2 + self.Ri**2) + (1/12) * self.mr3 * self.L3**2
        
    def _compute_gravity_coefficients(self):
        """预计算重力势能的系数"""
        self.K = (self.m * (-self.d1 + 3*self.d1p) + 
                  self.mr1 * (-self.d1 + self.d1p)/2 + 
                  self.mr2 * self.d1p + 
                  self.mr3 * self.d1p + 
                  2 * self.mB * self.d1p)
        
        self.L_coeff = (self.m * (-self.d2 + 2*self.d2p) + 
                        self.mr2 * (-self.d2 + self.d2p)/2 + 
                        self.mr3 * self.d2p + 
                        self.mB * self.d2p)
    
    def mass_matrix(self, q):
        """计算质量矩阵 M(q)"""
        theta1, phi2, phi3 = q
        
        # 基于MATLAB代码中的动能表达式构建质量矩阵
        M = np.zeros((3, 3))
        
        # M11: θ1的平方项系数
        M[0,0] = (self.mr1 * ((self.d1-self.d1p)/2)**2 + 
                  self.mr2 * (self.d1**2 + ((self.d2-self.d2p)/2)**2) +
                  self.mr3 * (self.d1p**2 + self.d2p**2) +
                  self.mB * (self.d1p**2 + self.d2p**2 + 1 + self.d2**2) +
                  self.m * self.d1**2 +  # 集中质量1贡献
                  self.Iperp1 + self.Iperp2 + self.Iperp3)
        
        # M22: φ2的平方项系数
        M[1,1] = (self.mr2 * ((self.d2-self.d2p)/2)**2 +
                  self.mB * self.d2**2 +
                  self.m * self.d2**2 +  # 集中质量2贡献
                  self.Ipar1 + self.Iperp2 + self.Iperp3)
        
        # M33: φ3的平方项系数  
        M[2,2] = self.Iperp2 + self.Iperp3
        
        # 耦合项 M12, M13, M23（考虑非线性耦合）
        cos_phi2 = np.cos(phi2)
        sin_phi2 = np.sin(phi2)
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        
        # M12 = M21: θ1和φ2的耦合
        M[0,1] = M[1,0] = 0.5 * self.m * self.L3**2
        
        # M13 = M31: θ1和φ3的耦合
        M[0,2] = M[2,0] = (self.Iperp2 * cos_phi2 + 
                           (self.Ipar2 - self.Iperp2) * cos_phi2)
        
        # M23 = M32: φ2和φ3的耦合
        M[1,2] = M[2,1] = (-2 * self.Iperp2 * sin_theta1 * cos_theta1 * sin_phi2)
        
        return M
    
    def coriolis_vector(self, q, dq):
        """计算科氏力向量 C(q,dq)"""
        theta1, phi2, phi3 = q
        dtheta1, dphi2, dphi3 = dq
        
        C = np.zeros(3)
        
        # 基于克氏符号的严格计算（简化版本，保留主要非线性项）
        sin_phi2 = np.sin(phi2)
        cos_phi2 = np.cos(phi2)
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        
        # C1: θ1方程的科氏力项
        C[0] = (self.Iperp2 * (-sin_phi2) * dphi2 * dtheta1 +
                2 * self.Iperp2 * sin_phi2 * dphi2 * dphi3 * cos_theta1 * sin_theta1 -
                self.Iperp2 * sin_phi2 * dphi2 * dphi3)
        
        # C2: φ2方程的科氏力项
        C[1] = (0.5 * self.Iperp2 * sin_phi2 * dtheta1**2 +
                self.Iperp2 * (cos_theta1**2 - sin_theta1**2) * sin_phi2 * dphi3**2 +
                2 * self.Iperp2 * cos_theta1 * sin_theta1 * cos_phi2 * dtheta1 * dphi3)
        
        # C3: φ3方程的科氏力项  
        C[2] = (-self.Iperp2 * sin_phi2 * dphi2 * dtheta1 -
                2 * self.Iperp2 * cos_theta1 * sin_theta1 * cos_phi2 * dphi2 * dtheta1 +
                2 * self.Iperp2 * cos_theta1 * sin_theta1 * sin_phi2 * dphi2**2)
        
        return C
    
    def gravity_vector(self, q):
        """计算重力向量 G(q)"""
        theta1, phi2, phi3 = q
        
        G = np.zeros(3)
        
        # 基于势能V = g*(K*cos(theta1) + L*sin(theta1)*cos(phi2))的梯度
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        sin_phi2 = np.sin(phi2)
        cos_phi2 = np.cos(phi2)
        
        # dV/dtheta1
        G[0] = self.g * (-self.K * sin_theta1 + self.L_coeff * cos_theta1 * cos_phi2)
        
        # dV/dphi2  
        G[1] = self.g * (-self.L_coeff * sin_theta1 * sin_phi2)
        
        # dV/dphi3 = 0 (势能中无φ3项)
        G[2] = 0
        
        return G
    
    def equations_of_motion(self, t, state):
        """严格的拉格朗日运动方程"""
        q = state[:3]  # [theta1, phi2, phi3]
        dq = state[3:]  # [dtheta1, dphi2, dphi3]
        
        # 计算各项
        M = self.mass_matrix(q)
        C = self.coriolis_vector(q, dq)
        G = self.gravity_vector(q)
        
        # 阻尼力
        tau_damping = -np.diag([self.b1, self.b2, self.b3]) @ dq
        
        # 求解 M * ddq = -C - G + tau
        try:
            ddq = np.linalg.solve(M, -C - G + tau_damping)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            ddq = np.linalg.pinv(M) @ (-C - G + tau_damping)
        
        return np.concatenate([dq, ddq])
    
    def compute_energy(self, state):
        """计算系统总能量，基于MATLAB代码中的能量表达式"""
        q = state[:3]
        dq = state[3:]
        
        theta1, phi2, phi3 = q
        dtheta1, dphi2, dphi3 = dq
        
        # 根据MATLAB代码中energy_terms_sym函数计算动能
        # 平动动能（杆和集中质量）
        v1_sq = dtheta1**2 * ((self.d1 - self.d1p)/2)**2
        v2_sq = (dtheta1**2 * (self.d1**2 + ((self.d2 - self.d2p)/2)**2) + 
                 dphi2**2 * ((self.d2 - self.d2p)/2)**2)
        v3_sq = (dtheta1**2 * (self.d1p**2 + self.d2p**2) + 
                 self.d2p**2 * dphi2**2)
        
        T_trans = (0.5 * self.mr1 * v1_sq + 
                   0.5 * self.mr2 * v2_sq + 
                   0.5 * self.mr3 * v3_sq +
                   self.mB * ((self.d1p**2 + self.d2p**2) * dtheta1**2 + 
                             dtheta1**2 + self.d2**2 * dphi2**2))
        
        # 集中质量的贡献
        T_trans += 0.5 * self.m * dtheta1**2 * self.d1**2  # 质量在第一段
        T_trans += 0.5 * self.m * (dtheta1**2 * (self.d1p**2 + self.d2**2) + 
                                   dphi2**2 * self.d2**2)  # 质量在第二段
        T_trans += 0.5 * self.m * (2 * dtheta1**2 * (self.d1p**2 + self.d2p**2) + 
                                   0.5 * self.L3**2 * (dtheta1**2 + 2 * dphi2**2) + 
                                   dtheta1 * dphi2 * self.L3**2)  # 质量在第三段
        
        # 转动动能
        T_rot1 = 0.5 * self.Iperp1 * dtheta1**2 + 0.5 * self.Ipar1 * dphi2**2
        
        cos_phi2 = np.cos(phi2)
        sin_phi2 = np.sin(phi2)
        sin_theta1 = np.sin(theta1)
        cos_theta1 = np.cos(theta1)
        
        T_rot2 = (0.5 * self.Iperp2 * (dtheta1**2 + dphi2**2 + dphi3**2 + 
                                       2 * dtheta1 * dphi3 * cos_phi2 - 
                                       4 * dphi2 * dphi3 * sin_theta1 * cos_theta1 * sin_phi2) +
                  0.5 * (self.Ipar2 - self.Iperp2) * (dtheta1 * cos_phi2 + dphi3)**2)
        
        T_rot3 = 0.5 * self.Iperp3 * (dtheta1**2 + dphi2**2 + dphi3**2 + 
                                      2 * dtheta1 * dphi3 * cos_phi2 - 
                                      4 * dphi2 * dphi3 * sin_theta1 * cos_theta1 * sin_phi2)
        
        T = T_trans + T_rot1 + T_rot2 + T_rot3
        
        # 势能 V（修正后的表达式）
        V = self.g * (self.K * cos_theta1 + self.L_coeff * sin_theta1 * cos_phi2)
        
        return T + V
    
    def lyapunov_exponent(self, initial_state, dt=0.1, total_time=100, n_vectors=3):
        """计算最大李雅普诺夫指数"""
        eps = 1e-8
        
        # 初始化
        state = np.array(initial_state)
        perturbations = eps * np.random.randn(n_vectors, 6)
        
        lyap_sum = 0
        n_steps = 0
        
        t_span = np.arange(0, total_time, dt)
        
        for i in range(len(t_span)-1):
            t_current = t_span[i]
            t_next = t_span[i+1]
            
            # 积分主轨道
            sol = solve_ivp(self.equations_of_motion, [t_current, t_next], 
                          state, dense_output=False, rtol=1e-9, atol=1e-11)
            state = sol.y[:, -1]
            
            # 积分扰动轨道
            perturbed_states = []
            for j in range(n_vectors):
                perturbed_initial = initial_state + perturbations[j]
                sol_pert = solve_ivp(self.equations_of_motion, [t_current, t_next],
                                   perturbed_initial, dense_output=False, 
                                   rtol=1e-9, atol=1e-11)
                perturbed_states.append(sol_pert.y[:, -1])
            
            # 计算偏差向量
            deviations = []
            for j in range(n_vectors):
                dev = perturbed_states[j] - state
                deviations.append(dev)
            
            # Gram-Schmidt正交化
            orthogonal_vectors = []
            for j in range(n_vectors):
                vec = deviations[j].copy()
                for k in range(j):
                    vec -= np.dot(vec, orthogonal_vectors[k]) * orthogonal_vectors[k]
                
                norm_vec = norm(vec)
                if norm_vec > 1e-12:
                    orthogonal_vectors.append(vec / norm_vec)
                    if j == 0:  # 只取最大的
                        lyap_sum += np.log(norm_vec / eps)
                        n_steps += 1
                else:
                    orthogonal_vectors.append(np.random.randn(6))
                    orthogonal_vectors[-1] /= norm(orthogonal_vectors[-1])
            
            # 更新扰动
            for j in range(n_vectors):
                perturbations[j] = eps * orthogonal_vectors[j]
            
            initial_state = state.copy()
        
        return lyap_sum / (n_steps * dt) if n_steps > 0 else 0
    
    def simulate(self, initial_state, t_span, **kwargs):
        """仿真系统"""
        sol = solve_ivp(self.equations_of_motion, [t_span[0], t_span[-1]], 
                       initial_state, t_eval=t_span, 
                       rtol=kwargs.get('rtol', 1e-9),
                       atol=kwargs.get('atol', 1e-11),
                       method=kwargs.get('method', 'DOP853'))
        return sol
    
    def find_chaos_region(self, theta1_range=(0.01, 3.13), n_points=20, **kwargs):
        """扫描混沌区域"""
        theta1_list = np.linspace(theta1_range[0], theta1_range[1], n_points)
        results = {'theta1': [], 'lyap_exp': [], 'is_chaotic': []}
        
        # 基线计算（用于确定混沌阈值）
        baseline_state = [0.02, 0, 0, 0, 0, 0]
        baseline_lyap = []
        for _ in range(4):
            lyap = self.lyapunov_exponent(baseline_state, **kwargs)
            baseline_lyap.append(lyap)
        
        threshold = np.mean(baseline_lyap) + 3 * np.std(baseline_lyap)
        
        print(f"混沌阈值: {threshold:.6f}")
        
        for theta1_0 in theta1_list:
            # 添加小扰动
            initial_state = [theta1_0, 1e-4*np.random.randn(), 1e-4*np.random.randn(), 0, 0, 0]
            
            lyap = self.lyapunov_exponent(initial_state, **kwargs)
            is_chaotic = lyap > threshold
            
            results['theta1'].append(theta1_0)
            results['lyap_exp'].append(lyap)
            results['is_chaotic'].append(is_chaotic)
            
            status = "混沌" if is_chaotic else "非混沌"
            print(f"θ1={theta1_0:.3f}: λ_max={lyap:.6f} ({status})")
        
        return results

# 使用示例
if __name__ == "__main__":
    # 创建系统
    system = LagrangianChaoticSystem()
    
    # 测试初始条件
    initial_state = [0.5, 0.1, 0.05, 0, 0, 0]  # [theta1, phi2, phi3, dtheta1, dphi2, dphi3]
    
    # 短时间仿真测试
    t_span = np.linspace(0, 10, 1000)
    sol = system.simulate(initial_state, t_span)
    
    print("系统创建成功！")
    print(f"初始能量: {system.compute_energy(initial_state):.6f}")
    print(f"最终能量: {system.compute_energy(sol.y[:, -1]):.6f}")
    
    # 计算李雅普诺夫指数（小规模测试）
    print("\n计算李雅普诺夫指数...")
    lyap = system.lyapunov_exponent(initial_state, total_time=20)
    print(f"最大李雅普诺夫指数: {lyap:.6f}")
    
    if lyap > 0:
        print("系统展现混沌行为！")
    else:
        print("系统在此初始条件下表现为规律运动")