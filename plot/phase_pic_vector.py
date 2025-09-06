
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from plot_traj import load_ball_tracks, merge_3d_tracks
import numpy as np
from scipy.signal import savgol_filter
from itertools import combinations

Width = 640
Height = 480


def interpolate_data(tracks_3d, new_sampling_rate=10):
    """对数据进行插值处理，提高采样率"""
    interpolated_tracks = {}
    
    for cname, points in tracks_3d.items():
        if not points:
            continue
        
        # 分别提取 x, y, z 坐标
        xs, ys, zs = zip(*points)
        
        # 假设原始数据的时间间隔是均匀的，可以用时间索引代替时间戳
        t_original = np.arange(len(xs))  # 原始时间索引
        t_new = np.linspace(0, len(xs)-1, len(xs) * new_sampling_rate)  # 插值后的时间索引
        
        # 使用样条插值进行数据插值
        cs_x = CubicSpline(t_original, xs)
        cs_y = CubicSpline(t_original, ys)
        cs_z = CubicSpline(t_original, zs)
        
        # 获取插值后的数据
        xs_new = cs_x(t_new)
        ys_new = cs_y(t_new)
        zs_new = cs_z(t_new)
        
        # 存储插值后的数据
        interpolated_tracks[cname] = list(zip(xs_new, ys_new, zs_new))
    
    return interpolated_tracks

def plot_interpolated_3d(tracks_3d, cname="v1"):
    """绘制插值后的三维轨迹"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if cname in tracks_3d:
        xs, ys, zs = zip(*tracks_3d[cname])
        ax.plot(xs, ys, zs, label=cname)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title(f"Interpolated 3D Trajectory: {cname}")

def plot_phase_space_interpolated(angles, velocities, cname="v1"):
    """绘制插值后的相空间图：角度 vs 角速度"""
    if cname not in angles or len(angles[cname]) == 0:
        print(f"警告: {cname} 的角度数据为空，跳过相空间图绘制")
        return
        
    theta, phi = angles[cname]
    omega_theta, omega_phi = velocities[cname]
    
    plt.figure(figsize=(8, 6))
    plt.plot(theta, omega_theta, '.', markersize=1, label=f'{cname} - θ vs ω_θ')
    plt.xlabel('θ (Angle)')
    plt.ylabel('ω_θ (Angular Velocity)')
    plt.title(f"Phase Space: {cname}")
    plt.legend()

def plot_delay_embedding_interpolated(angles, cname="v1", tau=1, m=9):
    """绘制插值后的延迟嵌入图"""
    if cname not in angles or len(angles[cname]) == 0:
        print(f"警告: {cname} 的角度数据为空，跳过延迟嵌入图绘制")
        return
        
    theta, phi = angles[cname]

    T = len(theta) - (m-1)*tau
    if T <= 0:
        print(f"警告: {cname} 的数据长度不足以进行延迟嵌入，跳过")
        return
        
    emb = np.zeros((T, m))
    for i in range(m):
        emb[:, i] = theta[i*tau : i*tau + T]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(emb[:,0], emb[:,1], emb[:,2], lw=0.5)
    ax.set_xlabel('θ(t)')
    ax.set_ylabel(f'θ(t+{tau})')
    ax.set_zlabel(f'θ(t+{2*tau})')
    plt.title(f'Delay Embedding (3D): {cname}')

def plot_poincare_section_interpolated(angles, velocities, cname="v1"):
    """绘制插值后的Poincaré截面图"""
    if cname not in angles or len(angles[cname]) == 0:
        print(f"警告: {cname} 的角度数据为空，跳过Poincaré截面图绘制")
        return
        
    # theta, phi = angles[cname]
    phi, theta = angles[cname]
    omega_theta, omega_phi = velocities[cname]
    
    cross_idx = np.where((theta[:-1] < 0.1) & (theta[1:] >= -0.1))[0]
    
    if len(cross_idx) == 0:
        print(f"未检测到过零点：{cname}")
        return  # 如果没有过零点，则直接退出
    
    # 用插值精确找到过零点
    poincare_theta = []
    poincare_omega = []
    
    for idx in cross_idx:
        # 线性插值计算过零点位置
        t1, t2 = theta[idx], theta[idx + 1]
        dt1, dt2 = omega_theta[idx], omega_theta[idx + 1]
        
        # 线性插值过零点
        zero_cross_theta = t1 - t2 / (dt2 - dt1) * (t2 - t1)
        zero_cross_omega = dt1 + (zero_cross_theta - t1) * (dt2 - dt1) / (t2 - t1)
        
        poincare_theta.append(zero_cross_theta)
        poincare_omega.append(zero_cross_omega)
    
    # 绘制Poincaré截面图
    plt.figure(figsize=(8, 6))
    plt.plot(poincare_theta, poincare_omega, '.', markersize=2)
    plt.xlabel('θ (Angle) at section')
    plt.ylabel('ω (Angular Velocity) at section')
    plt.title(f'Poincaré Section: {cname}')
    plt.grid(True)

def central_diff(x, dt):
    """计算中心差分的角速度"""
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

def calculate_angles_and_velocities(tracks_3d, dt, normalize:bool=True, L1=0.68, L2=0.32):
    """计算三维轨迹对应的角度和角速度"""
    angles = {}
    velocities = {}

    # 检查绿球和蓝球轨迹是否存在
    if "green_ball" not in tracks_3d or "blue_ball" not in tracks_3d:
        print("警告: 缺少绿球或蓝球轨迹数据")
        return angles, velocities
        
    if not tracks_3d["green_ball"] or not tracks_3d["blue_ball"]:
        print("警告: 绿球或蓝球轨迹数据为空")
        return angles, velocities
    red_track = tracks_3d["red_ball"] 
    green_track = tracks_3d["green_ball"]
    blue_track = tracks_3d["blue_ball"]
    
    # 找到两个轨迹的最小长度，进行对齐
    min_length = min(len(green_track), len(blue_track), len(red_track))
    
    if min_length == 0:
        print("警告: 轨迹数据长度为0")
        return angles, velocities
    
    # 截取到相同长度
    green_track = green_track[:min_length]
    blue_track = blue_track[:min_length]
    red_track = red_track[:min_length]
    
    xg, yg, zg = zip(*green_track)
    xb, yb, zb = zip(*blue_track)
    xr, yr, zr = zip(*red_track)

    xo, yo, zo = Width / 2, Height / 2, 0  # 假设观察点在图像中心，z=0平面

    xm1 = (np.array(xb) + np.array(xg))/2
    ym1 = (np.array(yb) + np.array(yg))/2
    zm1 = (np.array(zb) + np.array(zg))/2

    xm2 = (L1* np.array(xr) + L2* np.array(xm1))/(L1+L2)
    ym2 = (L1* np.array(yr) + L2* np.array(ym1))/(L1+L2)
    zm2 = (L1* np.array(zr) + L2* np.array(zm1))/(L1+L2) 



    


    # 绿球指向蓝球的向量
    xs1 = np.array(xb) - np.array(xg)  
    ys1 = np.array(yb) - np.array(yg)
    zs1 = np.array(zb) - np.array(zg)

    xs2 = np.array(xm1) - np.array(xr)
    ys2 = np.array(ym1) - np.array(yr)
    zs2 = np.array(zm1) - np.array(zr)

    xs3 = np.array(xm2) - xo
    ys3 = np.array(ym2) - yo
    zs3 = np.array(zm2) - zo


    #计算theta1
    vo = np.array([0, 1, 0])  # 观察点指向y轴正方向的单位向量

    s3 = np.stack([xs3, ys3, zs3], axis=1)
    s3_abs = np.linalg.norm(s3, axis=1, keepdims=True)
    s3_unit = s3 / s3_abs  # 归一化
    vo_abs = np.linalg.norm(vo)
    vo_unit = vo / vo_abs  # 归一化

    cos_angles = np.clip(np.dot(s3_unit, vo_unit), -1.0, 1.0)
    theta1 = np.arccos(cos_angles)  # shape: (N,)
    omega_theta1 = central_diff(theta1, dt)
    
    # 计算phi2：首帧为0，后续为相邻帧s2夹角累加
    s2 = np.stack([xs2, ys2, zs2], axis=1)  # (N, 3)
    N = s2.shape[0]
    phi2 = np.zeros(N)
    for i in range(1, N):
        v1 = s2[i-1] / np.linalg.norm(s2[i-1])
        v2 = s2[i] / np.linalg.norm(s2[i])
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # 判断方向，使用叉乘与s3方向判断正负
        s3_dir = np.array([xs3[i], ys3[i], zs3[i]])
        s3_dir = s3_dir / np.linalg.norm(s3_dir)
        cross = np.cross(v1, v2)
        sign = np.sign(np.dot(cross, s3_dir))
        phi2[i] = phi2[i-1] + angle * sign
    omega_phi2 = central_diff(phi2, dt)


    # 计算phi3：首帧为0，后续为相邻帧s3夹角累加，带方向
    s3 = np.stack([xs3, ys3, zs3], axis=1)  # (N, 3)
    N = s3.shape[0]
    phi3 = np.zeros(N)
    for i in range(1, N):
        v1 = s3[i-1] / np.linalg.norm(s3[i-1])
        v2 = s3[i] / np.linalg.norm(s3[i])
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # 判断方向，使用叉乘与s1方向判断正负
        s1_dir = np.array([xs1[i], ys1[i], zs1[i]])
        s1_dir = s1_dir / np.linalg.norm(s1_dir)
        cross = np.cross(v1, v2)
        sign = np.sign(np.dot(cross, s1_dir))
        phi3[i] = phi3[i-1] + angle * sign
    omega_phi3 = central_diff(phi3, dt)


    # 只计算存在数据的向量
    angles["theta1"] = theta1
    velocities["theta1"] = omega_theta1

    angles["phi2"] = phi2
    velocities["phi2"] = omega_phi2

    angles["phi3"] = phi3
    velocities["phi3"] = omega_phi3



    return angles, velocities

def filter_and_normalize_tracks(tracks_3d, window_length=11, polyorder=3):
    """对每个球的x, y, z序列进行Savitzky-Golay滤波和归一化"""
    filtered_tracks = {}
    for cname, points in tracks_3d.items():
        if not points:
            continue
        xs, ys, zs = zip(*points)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        # 滤波
        if len(xs) >= window_length:
            xs_f = savgol_filter(xs, window_length, polyorder)
            ys_f = savgol_filter(ys, window_length, polyorder)
            zs_f = savgol_filter(zs, window_length, polyorder)
        else:
            xs_f, ys_f, zs_f = xs, ys, zs
        # 归一化
        xs_f = (xs_f - np.mean(xs_f)) / np.std(xs_f) if np.std(xs_f) > 0 else xs_f
        ys_f = (ys_f - np.mean(ys_f)) / np.std(ys_f) if np.std(ys_f) > 0 else ys_f
        zs_f = (zs_f - np.mean(zs_f)) / np.std(zs_f) if np.std(zs_f) > 0 else zs_f
        filtered_tracks[cname] = list(zip(xs_f, ys_f, zs_f))
    return filtered_tracks

def plot_all_phase_spaces(angles, velocities, names):
    """
    绘制三组变量的相空间图（角度-角速度），三幅子图在一个figure中。
    angles, velocities: dict，key为变量名，value为(θ, φ)或(ω_θ, ω_φ)
    names: 长度为3的变量名列表，如['v1', 'v2', 'v3']
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, name in enumerate(names):
        if name not in angles or name not in velocities:
            axs[i].set_title(f"{name} 不存在")
            continue
        theta = angles[name]
        omega_theta = velocities[name]
        axs[i].plot(theta, omega_theta, '.', markersize=1)
        axs[i].set_xlabel('θ (Angle)')
        axs[i].set_ylabel('ω_θ (Angular Velocity)')
        axs[i].set_title(f"Phase Space: {name}")
    plt.tight_layout()


def plot_all_3d_phase_spaces(angles, velocities):
    """
    绘制angles和velocities中三个角度和三个角速度的20种三元组三维相图。
    顺序、轴标注与attractor_model.py的plot_enhanced_attractors一致。
    """
    

    # 变量名和标签，顺序与仿真一致
    var_names = [r'$\\theta_1$', r'$\\varphi_2$', r'$\\varphi_3$', r'$d\\theta_1/dt$', r'$d\\varphi_2/dt$', r'$d\\varphi_3/dt$']
    var_labels = ['θ₁ (rad)', 'φ₂ (rad)', 'φ₃ (rad)', 'dθ₁/dt (rad/s)', 'dφ₂/dt (rad/s)', 'dφ₃/dt (rad/s)']
    # angles/velocities的顺序
    angle_keys = ['theta1', 'phi2', 'phi3']
    vel_keys = ['theta1', 'phi2', 'phi3']
    # 组装数据
    data = [angles.get('theta1', None), angles.get('phi2', None), angles.get('phi3', None),
            velocities.get('theta1', None), velocities.get('phi2', None), velocities.get('phi3', None)]
    # 组合
    n_vars = 6
    combs = list(combinations(range(n_vars), 3))  # 20种三元组
    # 颜色
    color = '#FF6F61'  # 柔和橙红色
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
            # 检查数据有效性
            if data[i] is None or data[j] is None or data[k] is None:
                ax.set_title('数据缺失')
                continue
            N = min(len(data[i]), len(data[j]), len(data[k]))
            ax.plot(data[i][:N], data[j][:N], data[k][:N], color=color, alpha=0.95, linewidth=0.8)
            ax.set_xlabel(var_labels[i])
            ax.set_ylabel(var_labels[j])
            ax.set_zlabel(var_labels[k])
            ax.set_title(f'3D-{var_labels[i]}-{var_labels[j]}-{var_labels[k]}')
        fig.suptitle(f'fig2_{fig_idx+1}: 6D-3D({fig_idx*4+1}-{min((fig_idx+1)*4,20)})', fontsize=16)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

if __name__ == "__main__":
    track_dir = ".\\traces_tiny"
    tracks1 = load_ball_tracks(track_dir, 1)
    tracks2 = load_ball_tracks(track_dir, 2)

    # 合并两个轨迹
    tracks_3d = merge_3d_tracks(tracks1, tracks2)

    filtered_tracks_3d = filter_and_normalize_tracks(tracks_3d, window_length=19, polyorder=6)
    # 插值处理
    interpolated_tracks_3d = interpolate_data(tracks_3d, new_sampling_rate=10)  # 将采样率提高10倍
    
    # 绘制插值后的相空间图和其他可视化
    angles, velocities = calculate_angles_and_velocities(filtered_tracks_3d, dt=1/60,normalize=False)  # 采样率为60Hz
    
    plot_all_phase_spaces(angles, velocities, ['theta1', 'phi2', 'phi3'])

    plot_all_3d_phase_spaces(angles, velocities)

    
    plt.show()