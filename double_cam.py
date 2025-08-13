import cv2
import threading
import queue
from ball_track_yolo import RGBBallTracker, save_ball_tracks, update_ball_tracks
import torch
import os


"""
global variable
"""
thread_running_event = threading.Event()

# 摄像头帧队列
frame_queue_0 = queue.Queue(maxsize=5)
frame_queue_1 = queue.Queue(maxsize=5)

# 摄像头采集线程

def cam_reader(cam_id, frame_queue, thread_running_event:threading.Event):
    cap = cv2.VideoCapture(cam_id)
     # 摄像头优化设置
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟

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





if __name__ == "__main__":

    modle_path = "ball.pt"

    print("🚀 RGB球追踪模型测试-双摄像头同步")
    print("=" * 50)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠ 未检测到GPU，将使用CPU")

    isPrintInfo = input('是否绘制统计信息？(y/n): ')
    if isPrintInfo.lower() == 'y':
        print("✓ 将绘制统计信息")
        isPrintInfo = True
    else:
        print("✗ 不绘制统计信息")
        isPrintInfo = False

    # 启动摄像头线程
    t0 = threading.Thread(target=cam_reader, args=(0, frame_queue_0, thread_running_event), daemon=True)
    t1 = threading.Thread(target=cam_reader, args=(2, frame_queue_1, thread_running_event), daemon=True)
    t0.start()
    t1.start()

    # 加载YOLO模型
    tracker1 = RGBBallTracker(modle_path)
    tracker2 = RGBBallTracker(modle_path)

    # 轨迹记录相关
    tracking = False
    ball_tracks1 = {'red_ball': [], 'green_ball': [], 'blue_ball': []}
    ball_tracks2 = {'red_ball': [], 'green_ball': [], 'blue_ball': []}
    print("按 s 开始/暂停追踪，按 q 退出并保存轨迹")

    while True:
        frame0 = frame_queue_0.get()
        frame1 = frame_queue_1.get()
        processed0, det0 = tracker1.process_frame(frame0, isPrintInfo=isPrintInfo)
        processed1, det1 = tracker2.process_frame(frame1, isPrintInfo=isPrintInfo)

        # 追踪并记录轨迹
        if tracking:
            update_ball_tracks(ball_tracks1, det0)
            update_ball_tracks(ball_tracks2, det1)

        # 显示
        cv2.imshow('cam0', processed0)
        cv2.imshow('cam1', processed1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            tracking = not tracking
            print("追踪" + ("开始" if tracking else "暂停"))
        elif key == ord('q'):
            thread_running_event.set()
            print("退出程序，保存轨迹...")
            break

    trace_save_dir = ".\\traces"
    save_ball_tracks(ball_tracks1,trace_save_dir,1)
    save_ball_tracks(ball_tracks2,trace_save_dir,2)
    cv2.destroyAllWindows()
