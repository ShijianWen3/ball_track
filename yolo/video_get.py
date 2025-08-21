import cv2
import time
import os
from datetime import datetime
import threading
import queue


"""
亮度:20


"""



class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.video_writer = None
        self.frame_queue = queue.Queue(maxsize=300)  # 缓冲队列
        self.recording_thread = None
        self.total_frames = 0
        self.dropped_frames = 0
        self.record_start_time = None
        
        # 视频保存设置
        self.output_dir = "captured_videos"
        self.current_filename = None
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
    
    def start_recording(self, width, height, fps):
        """开始录制"""
        if self.recording:
            print("已在录制中...")
            return False
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"pendulum_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, self.current_filename)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (int(width), int(height)))
        
        if not self.video_writer.isOpened():
            print("无法创建视频文件!")
            return False
        
        self.recording = True
        self.total_frames = 0
        self.dropped_frames = 0
        self.record_start_time = time.time()
        
        # 启动录制线程
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"开始录制: {filepath}")
        print(f"分辨率: {int(width)}x{int(height)}, FPS: {fps}")
        return True
    
    def _recording_worker(self):
        """录制工作线程"""
        while self.recording:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    self.video_writer.write(frame)
                    self.total_frames += 1
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"录制线程错误: {e}")
                break
    
    def add_frame(self, frame):
        """添加帧到录制队列"""
        if not self.recording:
            return
        
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped_frames += 1
            # 清理一些旧帧为新帧腾出空间
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
            except:
                pass
    
    def stop_recording(self):
        """停止录制"""
        if not self.recording:
            return
        
        self.recording = False
        
        # 等待队列处理完成
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        
        # 释放视频写入器
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # 计算录制统计信息
        if self.record_start_time:
            record_duration = time.time() - self.record_start_time
            actual_fps = self.total_frames / record_duration if record_duration > 0 else 0
            
            print(f"录制完成: {self.current_filename}")
            print(f"录制时长: {record_duration:.2f}秒")
            print(f"总帧数: {self.total_frames}")
            print(f"丢失帧数: {self.dropped_frames}")
            print(f"实际录制FPS: {actual_fps:.2f}")
            
            # 显示文件信息
            filepath = os.path.join(self.output_dir, self.current_filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"文件大小: {file_size:.2f} MB")

def draw_recording_overlay(frame, recorder, fps, frame_count, elapsed_time):
    """绘制录制状态覆盖层"""
    height, width = frame.shape[:2]
    
    # 录制状态指示器
    if recorder.recording:
        # 录制中 - 红色圆点闪烁
        blink = int(time.time() * 2) % 2
        if blink:
            cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 70, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 录制时间
        record_time = time.time() - recorder.record_start_time
        minutes = int(record_time // 60)
        seconds = int(record_time % 60)
        cv2.putText(frame, f"{minutes:02d}:{seconds:02d}", (width - 100, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 录制帧数和丢失帧数
        cv2.putText(frame, f"Frames: {recorder.total_frames}", (width - 150, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if recorder.dropped_frames > 0:
            cv2.putText(frame, f"Dropped: {recorder.dropped_frames}", (width - 150, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # FPS和帧数信息
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 缓冲区状态
    if recorder.recording:
        queue_size = recorder.frame_queue.qsize()
        max_queue = recorder.frame_queue.maxsize
        queue_percent = (queue_size / max_queue) * 100
        
        color = (0, 255, 0)  # 绿色
        if queue_percent > 80:
            color = (0, 0, 255)  # 红色
        elif queue_percent > 60:
            color = (0, 165, 255)  # 橙色
        
        cv2.putText(frame, f"Buffer: {queue_size}/{max_queue} ({queue_percent:.0f}%)", 
                   (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    # 摄像头设置 (与追踪程序完全一致)
    camera_index = 1  # 你可以修改为你的摄像头索引
    
    # 打开摄像头，使用DirectShow后端（Windows推荐）
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        exit()

    # 设置640x480分辨率 (与追踪程序一致)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 设置120fps (与追踪程序一致)
    cap.set(cv2.CAP_PROP_FPS, 120)

    # 设置缓冲区大小（减少延迟）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 设置MJPG格式 (与追踪程序一致)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # 获取实际设置的参数
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"摄像头设置：{width}x{height} @ {actual_fps}fps")
    print("=" * 50)
    print("视频采集程序")
    print("按键说明：")
    print("SPACE - 开始/停止录制")
    print("q     - 退出程序")
    print("r     - 重置统计信息")
    print("i     - 显示/隐藏信息覆盖层")
    print("=" * 50)
    
    # 创建录制器
    recorder = VideoRecorder()
    
    # 初始化变量
    frame_count = 0
    start_time = time.time()
    fps = 0
    frame_times = []
    show_overlay = True
    
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取视频帧")
            break

        frame_count += 1
        elapsed_time = current_time - start_time
        
        # 记录帧时间用于FPS计算
        frame_times.append(current_time)
        frame_times = [t for t in frame_times if current_time - t < 1.0]
        fps = len(frame_times) if len(frame_times) > 1 else 0
        
        # 添加帧到录制队列
        recorder.add_frame(frame)
        
        # 绘制信息覆盖层
        if show_overlay:
            draw_recording_overlay(frame, recorder, fps, frame_count, elapsed_time)
        
        # 在底部添加按键提示
        if show_overlay:
            cv2.putText(frame, "SPACE: Record | Q: Quit | I: Toggle Info", 
                       (10, int(height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 显示画面
        cv2.imshow('Video Capture - Double Pendulum', frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键
            if recorder.recording:
                recorder.stop_recording()
            else:
                recorder.start_recording(width, height, actual_fps)
        elif key == ord('r'):
            # 重置统计
            frame_count = 0
            start_time = time.time()
            frame_times = []
            print("统计信息已重置")
        elif key == ord('i'):
            # 切换信息显示
            show_overlay = not show_overlay
            print(f"信息覆盖层: {'开启' if show_overlay else '关闭'}")

    # 停止录制
    if recorder.recording:
        print("正在停止录制...")
        recorder.stop_recording()

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    print(f"程序结束，最终FPS: {fps:.1f}")
    print(f"总共采集了 {frame_count} 帧")
    
    # 显示保存的视频文件
    if os.path.exists(recorder.output_dir):
        video_files = [f for f in os.listdir(recorder.output_dir) if f.endswith('.mp4')]
        if video_files:
            print(f"\n保存的视频文件 (在 {recorder.output_dir} 目录):")
            for i, filename in enumerate(video_files, 1):
                filepath = os.path.join(recorder.output_dir, filename)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"{i}. {filename} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main()