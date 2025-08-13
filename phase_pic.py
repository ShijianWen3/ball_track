import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from plot_traj import load_ball_tracks, merge_3d_tracks
import numpy as np
from scipy.signal import savgol_filter
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
def plot_interpolated_3d(tracks_3d, cname="red_ball"):
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

def plot_phase_space_interpolated(angles, velocities, cname="red_ball"):
    """绘制插值后的相空间图：角度 vs 角速度"""
    theta, phi = angles[cname]
    omega_theta, omega_phi = velocities[cname]
    
    plt.figure(figsize=(8, 6))
    plt.plot(theta, omega_theta, '.', markersize=1, label=f'{cname} - θ vs ω_θ')
    plt.xlabel('θ (Angle)')
    plt.ylabel('ω_θ (Angular Velocity)')
    plt.title(f"Phase Space: {cname}")
    plt.legend()

def plot_delay_embedding_interpolated(angles, cname="red_ball", tau=1, m=9):
    """绘制插值后的延迟嵌入图"""
    theta, phi = angles[cname]
    # phi, theta = angles[cname]

    T = len(theta) - (m-1)*tau
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

def plot_poincare_section_interpolated(angles, velocities, cname="red_ball"):
    """绘制插值后的Poincaré截面图"""
    theta, phi = angles[cname]
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


def calculate_angles_and_velocities(tracks_3d, dt, normalize=True):
    """计算三维轨迹对应的角度和角速度，并可选归一化"""
    angles = {"red_ball": [], "green_ball": [], "blue_ball": []}
    velocities = {"red_ball": [], "green_ball": [], "blue_ball": []}
    for cname in tracks_3d:
        xs, ys, zs = zip(*tracks_3d[cname])
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        phi = np.arctan2(ys, xs)
        theta = np.arctan2(zs, np.sqrt(xs**2 + ys**2))
        omega_phi = central_diff(phi, dt)
        omega_theta = central_diff(theta, dt)
        if normalize:
            theta = (theta - np.mean(theta)) / np.std(theta) if np.std(theta) > 0 else theta
            phi = (phi - np.mean(phi)) / np.std(phi) if np.std(phi) > 0 else phi
            omega_theta = (omega_theta - np.mean(omega_theta)) / np.std(omega_theta) if np.std(omega_theta) > 0 else omega_theta
            omega_phi = (omega_phi - np.mean(omega_phi)) / np.std(omega_phi) if np.std(omega_phi) > 0 else omega_phi
        angles[cname] = (theta, phi)
        velocities[cname] = (omega_theta, omega_phi)
    return angles, velocities


if __name__ == "__main__":
    track_dir = ".\\traces"
    tracks1 = load_ball_tracks(track_dir, 1)
    tracks2 = load_ball_tracks(track_dir, 2)
    # 合并两个轨迹
    tracks_3d = merge_3d_tracks(tracks1, tracks2)
    # 滤波+归一化
    filtered_tracks_3d = filter_and_normalize_tracks(tracks_3d, window_length=19, polyorder=6)
    # 插值处理（如需）
    processed_tracks_3d = interpolate_data(filtered_tracks_3d, new_sampling_rate=10)
    # processed_tracks_3d = filtered_tracks_3d
    # 计算角度和角速度（含归一化）
    angles, velocities = calculate_angles_and_velocities(processed_tracks_3d, dt=1/600, normalize=True)
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        plot_phase_space_interpolated(angles, velocities, cname)
        plot_delay_embedding_interpolated(angles, cname, tau=5, m=10)
        plot_poincare_section_interpolated(angles, velocities, cname)
    plt.show()
