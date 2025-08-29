
import cv2
import threading
import queue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import time
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from yolo.ball_track_yolo import RGBBallTracker, update_ball_tracks

# ========== 配置参数 ==========
MODEL_PATH = "ball.pt"
CAM_ID_0 = 0
CAM_ID_1 = 2
MAX_QUEUE_SIZE = 5
PLOT_REFRESH_INTERVAL = 0.05  # 秒

# ========== 全局变量 ==========
thread_running_event = threading.Event()
frame_queue_0 = queue.Queue(maxsize=MAX_QUEUE_SIZE)
frame_queue_1 = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def cam_reader(cam_id, frame_queue, thread_running_event:threading.Event):
	cap = cv2.VideoCapture(cam_id)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	if not cap.isOpened():
		print(f"无法打开摄像头{cam_id}")
		return
	while not thread_running_event.is_set():
		ret, frame = cap.read()
		if not ret:
			continue
		if not frame_queue.full():
			frame_queue.put(frame)
	cap.release()
	print(f"摄像头{cam_id}线程已停止")

def merge_3d_tracks_frame(det0, det1):
	# det0, det1: list of detection dicts
	# 返回: {class_name: (x0, y0, x1)}
	d3 = {}
	for cname in ["red_ball", "green_ball", "blue_ball"]:
		c0 = next((d for d in det0 if d['class_name']==cname), None)
		c1 = next((d for d in det1 if d['class_name']==cname), None)
		if c0 and c1:
			d3[cname] = (c0['center'][0], c0['center'][1], c1['center'][0])
	return d3

def main():
	print("🚀 双摄像头实时三维轨迹可视化")
	print("="*50)
	if torch.cuda.is_available():
		print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
	else:
		print("⚠ 未检测到GPU，将使用CPU")

	isPrintInfo = input('是否绘制统计信息？(y/n): ').lower() == 'y'

	# 启动摄像头线程
	t0 = threading.Thread(target=cam_reader, args=(CAM_ID_0, frame_queue_0, thread_running_event), daemon=True)
	t1 = threading.Thread(target=cam_reader, args=(CAM_ID_1, frame_queue_1, thread_running_event), daemon=True)
	t0.start()
	t1.start()

	# 加载YOLO模型
	tracker0 = RGBBallTracker(MODEL_PATH)
	tracker1 = RGBBallTracker(MODEL_PATH)

	# 轨迹记录，缓冲区长度
	TRAJ_BUF_LEN = 200
	tracks_3d = {"red_ball": [], "green_ball": [], "blue_ball": []}
	print("按 s 开始/暂停追踪，按 q 退出")
	tracking = False

	# 初始化3D点云绘图
	plt.ion()
	fig = plt.figure("3D Ball Point Cloud", figsize=(7, 6))
	ax = fig.add_subplot(111, projection='3d')
	color_map = {"red_ball": "r", "green_ball": "g", "blue_ball": "b"}

	def update_plot():
		ax.clear()
		for cname, points in tracks_3d.items():
			if len(points) > 0:
				xs, ys, zs = zip(*points)
				ax.scatter(xs, ys, zs, c=color_map.get(cname, "k"), alpha=0.8, label=cname)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	ax.set_title("3D Ball Point Cloud (实时)")
	plt.tight_layout()
	plt.draw()
	plt.pause(0.001)

	plt.show(block=False)

	try:
		while True:
			if frame_queue_0.empty() or frame_queue_1.empty():
				time.sleep(0.01)
				continue
			frame0 = frame_queue_0.get()
			frame1 = frame_queue_1.get()
			processed0, det0 = tracker0.process_frame(frame0, isPrintInfo=isPrintInfo)
			processed1, det1 = tracker1.process_frame(frame1, isPrintInfo=isPrintInfo)

			# 追踪并记录三维轨迹
			if tracking:
				d3 = merge_3d_tracks_frame(det0, det1)
				for cname, pt in d3.items():
					tracks_3d[cname].append(pt)
					if len(tracks_3d[cname]) > TRAJ_BUF_LEN:
						tracks_3d[cname].pop(0)

			# 显示摄像头画面
			cv2.imshow('cam0', processed0)
			cv2.imshow('cam1', processed1)

			# 实时刷新3D轨迹
			update_plot()

			key = cv2.waitKey(1) & 0xFF
			if key == ord('s'):
				tracking = not tracking
				print("追踪" + ("开始" if tracking else "暂停"))
			elif key == ord('q'):
				thread_running_event.set()
				print("退出程序...")
				break
			time.sleep(PLOT_REFRESH_INTERVAL)
	finally:
		plt.ioff()
		cv2.destroyAllWindows()
		print("程序已退出")

if __name__ == "__main__":
	main()

