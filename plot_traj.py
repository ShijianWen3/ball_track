import matplotlib.pyplot as plt
import os

def load_ball_tracks(track_dir="."):
    tracks = {}
    for cname in ["red_ball", "green_ball", "blue_ball"]:
        fname = os.path.join(track_dir, f"{cname}.txt")
        points = []
        if os.path.exists(fname):
            with open(fname, "r") as f:
                for line in f:
                    x, y = map(int, line.strip().split(","))
                    points.append((x, y))
        tracks[cname] = points
    return tracks

def plot_ball_tracks(tracks):
    color_map = {"red_ball": "r", "green_ball": "g", "blue_ball": "b"}
    plt.figure(figsize=(8, 6))
    for cname, points in tracks.items():
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", color=color_map.get(cname, "k"), label=cname)
    plt.legend()
    plt.title("Ball Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    tracks = load_ball_tracks()
    plot_ball_tracks(tracks)
