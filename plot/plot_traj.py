import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_ball_tracks(track_dir=".\\traces",suffix:int = 1):
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

def plot_ball_tracks(tracks, suffix:int = 1):
    """绘制球的轨迹"""
    color_map = {"red_ball": "r", "green_ball": "g", "blue_ball": "b"}
    plt.figure(figsize=(8, 6))
    for cname, points in tracks.items():
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", color=color_map.get(cname, "k"), label=cname)
    plt.legend()
    plt.title(f"Ball Trajectories {suffix}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()



def merge_3d_tracks(tracks1, tracks2):
    tracks_3d = {}
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        pts1 = tracks1.get(cname, [])
        pts2 = tracks2.get(cname, [])
        n = min(len(pts1), len(pts2))
        points_3d = [(pts1[i][0], pts1[i][1], pts2[i][0]) for i in range(n)]
        tracks_3d[cname] = points_3d
    return tracks_3d

def plot_single_ball_3d(points, cname):
    color_map = {"red_ball": "r", "green_ball": "g", "blue_ball": "b"}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points:
        xs, ys, zs = zip(*points)
        ax.scatter(xs, ys, zs, c=color_map.get(cname, "k"), label=cname)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.title(f"3D Trajectory: {cname}")

if __name__ == "__main__":
    track_dir = ".\\traces_13_8"
    tracks1 = load_ball_tracks(track_dir,1)
    tracks2 = load_ball_tracks(track_dir,2)
    plot_ball_tracks(tracks1, 1)
    plot_ball_tracks(tracks2, 2)
    
    tracks_3d = merge_3d_tracks(tracks1, tracks2)
    print(tracks_3d)
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        plot_single_ball_3d(tracks_3d[cname], cname)

    plt.show()
    
