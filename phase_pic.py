


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

def load_ball_tracks(track_dir=".\\traces", suffix:int = 1):
    tracks = {}
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        fname = os.path.join(track_dir, f"{cname}_{suffix}.txt")
        points = []
        if os.path.exists(fname):
            with open(fname, "r") as f:
                for line in f:
                    x, y = map(int, line.strip().split(","))
                    points.append((x, y))
        tracks[cname] = points
    return tracks

def merge_3d_tracks(tracks1, tracks2):
    tracks_3d = {}
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        pts1 = tracks1.get(cname, [])
        pts2 = tracks2.get(cname, [])
        n = min(len(pts1), len(pts2))
        points_3d = [(pts1[i][0], pts1[i][1], pts2[i][0]) for i in range(n)]
        tracks_3d[cname] = points_3d
    return tracks_3d

def central_diff(x, dt):
    """计算中心差分的角速度"""
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

def calculate_angles_and_velocities(tracks_3d, dt):
    """计算三维轨迹对应的角度和角速度"""
    angles = {"red_ball": [], "green_ball": [], "blue_ball": []}
    velocities = {"red_ball": [], "green_ball": [], "blue_ball": []}
    
    for cname in tracks_3d:
        # 从3D数据中提取角度与角速度
        xs, ys, zs = zip(*tracks_3d[cname])
        
        # 计算方位角和俯仰角
        phi = np.arctan2(ys, xs)
        theta = np.arctan2(zs, np.sqrt(np.array(xs)**2 + np.array(ys)**2))
        
        # 计算角速度
        omega_phi = central_diff(phi, dt)
        omega_theta = central_diff(theta, dt)
        
        angles[cname] = (theta, phi)
        velocities[cname] = (omega_theta, omega_phi)
    
    return angles, velocities

def plot_phase_space(angles, velocities, cname="red_ball"):
    """绘制相空间图：角度 vs 角速度"""
    theta, phi = angles[cname]
    omega_theta, omega_phi = velocities[cname]
    
    plt.figure(figsize=(8, 6))
    
    # 可以绘制 (theta1, omega_theta1) 或者 (theta1, theta2)
    plt.plot(theta, omega_theta, '.', markersize=1, label=f'{cname} - θ vs ω_θ')
    
    plt.xlabel('θ (Angle)')
    plt.ylabel('ω_θ (Angular Velocity)')
    plt.title(f"Phase Space: {cname}")
    plt.legend()

def plot_delay_embedding(angles, cname="red_ball", tau=1, m=3):
    """绘制延迟嵌入图，选择一个角度序列进行延迟嵌入"""
    theta, phi = angles[cname]
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

def plot_poincare_section(angles, velocities, cname="red_ball"):
    """绘制Poincaré截面：根据某个角度的过零点（例如θ3）"""
    theta, phi = angles[cname]
    omega_theta, omega_phi = velocities[cname]
    
    cross_idx = np.where((theta[:-1] < 0) & (theta[1:] >= 0))[0]
    poincare_theta = theta[cross_idx]
    poincare_omega = omega_theta[cross_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(poincare_theta, poincare_omega, '.', markersize=2)
    plt.xlabel('θ at section')
    plt.ylabel('ω at section')
    plt.title(f'Poincaré Section: {cname}')
    plt.show()

if __name__ == "__main__":
    track_dir = ".\\traces"
    tracks1 = load_ball_tracks(track_dir, 1)
    tracks2 = load_ball_tracks(track_dir, 2)

    tracks_3d = merge_3d_tracks(tracks1, tracks2)
    dt = 0.1  # 假设时间间隔为0.1秒，可以根据实际数据调整
    
    # 计算角度和角速度
    angles, velocities = calculate_angles_and_velocities(tracks_3d, dt)
    
    # 绘制相空间图、延迟嵌入和Poincaré截面
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        plot_phase_space(angles, velocities, cname)
        plot_delay_embedding(angles, cname, tau=1, m=3)
        plot_poincare_section(angles, velocities, cname)
    
    plt.show()
