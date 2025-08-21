import cv2
import numpy as np
import time
import math
from collections import deque
import matplotlib.pyplot as plt

class DoublePendulumTracker:
    def __init__(self):
        # 颜色范围定义 (HSV)
        # 红色球的HSV范围
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        # 绿色球的HSV范围
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
        
        # 轨迹存储 (保存最近N个点)
        self.red_trail = deque(maxlen=100)
        self.green_trail = deque(maxlen=100)
        
        # 位置和时间记录
        self.positions = deque(maxlen=1000)  # 存储位置和时间戳
        self.timestamps = deque(maxlen=1000)
        
        # 摆的物理参数估计
        self.pivot_point = None
        self.L1 = None  # 第一段摆长
        self.L2 = None  # 第二段摆长
        
        # 角度记录
        self.theta1_history = deque(maxlen=500)
        self.theta2_history = deque(maxlen=500)
        self.time_history = deque(maxlen=500)
        
        # 卡尔曼滤波器参数（简化版）
        self.red_filter = SimpleKalmanFilter()
        self.green_filter = SimpleKalmanFilter()
        
    def detect_ball(self, frame, color_name):
        """检测指定颜色的球"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if color_name == "red":
            # 红色需要两个范围
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # green
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓面积，过滤太小的
            area = cv2.contourArea(largest_contour)
            if area > 100:  # 最小面积阈值
                # 计算质心
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), largest_contour, area
        
        return None, None, 0
    
    def estimate_pivot_point(self, red_pos, green_pos):
        """估计摆的支点位置"""
        if red_pos is None or green_pos is None:
            return None
            
        # 简化假设：支点在两球连线的延长线上方
        # 这里需要根据实际情况调整
        x1, y1 = red_pos
        x2, y2 = green_pos
        
        # 假设支点在画面上方某个位置
        # 可以通过手动校准或几何分析确定
        if self.pivot_point is None:
            # 初始估计：在两球上方的中点
            pivot_x = (x1 + x2) // 2
            pivot_y = min(y1, y2) - 200  # 假设支点在上方200像素
            self.pivot_point = (pivot_x, pivot_y)
        
        return self.pivot_point
    
    def calculate_angles(self, pivot, red_pos, green_pos):
        """计算摆的角度"""
        if pivot is None or red_pos is None or green_pos is None:
            return None, None, None, None
        
        px, py = pivot
        r1x, r1y = red_pos
        r2x, r2y = green_pos
        
        # 计算第一段摆的角度（从支点到红球）
        L1 = math.sqrt((r1x - px)**2 + (r1y - py)**2)
        theta1 = math.atan2(r1x - px, r1y - py)
        
        # 计算第二段摆的角度（从红球到绿球）
        L2 = math.sqrt((r2x - r1x)**2 + (r2y - r1y)**2)
        theta2 = math.atan2(r2x - r1x, r2y - r1y) - theta1
        
        return theta1, theta2, L1, L2
    
    def draw_pendulum(self, frame, pivot, red_pos, green_pos):
        """绘制摆的结构"""
        if pivot is None or red_pos is None or green_pos is None:
            return
        
        # 绘制摆杆
        cv2.line(frame, pivot, red_pos, (255, 255, 255), 3)
        cv2.line(frame, red_pos, green_pos, (255, 255, 255), 3)
        
        # 绘制支点
        cv2.circle(frame, pivot, 8, (0, 0, 255), -1)
        
        # 绘制连接点
        cv2.circle(frame, red_pos, 5, (0, 255, 255), -1)
    
    def draw_trails(self, frame):
        """绘制轨迹"""
        # 绘制红球轨迹
        if len(self.red_trail) > 1:
            points = np.array(self.red_trail, dtype=np.int32)
            for i in range(1, len(points)):
                thickness = max(1, int(3 * i / len(points)))
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), (0, 0, 255), thickness)
        
        # 绘制绿球轨迹
        if len(self.green_trail) > 1:
            points = np.array(self.green_trail, dtype=np.int32)
            for i in range(1, len(points)):
                thickness = max(1, int(3 * i / len(points)))
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), (0, 255, 0), thickness)
    
    def draw_info(self, frame, red_pos, green_pos, theta1, theta2, L1, L2, fps):
        """绘制信息文本"""
        y_offset = 30
        
        # FPS信息
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # 位置信息
        if red_pos:
            cv2.putText(frame, f"Red Ball: ({red_pos[0]}, {red_pos[1]})", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
        
        if green_pos:
            cv2.putText(frame, f"Green Ball: ({green_pos[0]}, {green_pos[1]})", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # 角度信息
        if theta1 is not None and theta2 is not None:
            cv2.putText(frame, f"Theta1: {math.degrees(theta1):.1f}°", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            cv2.putText(frame, f"Theta2: {math.degrees(theta2):.1f}°", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        
        # 摆长信息
        if L1 is not None and L2 is not None:
            cv2.putText(frame, f"L1: {L1:.1f}px, L2: {L2:.1f}px", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

class SimpleKalmanFilter:
    """简化的卡尔曼滤波器用于位置平滑"""
    def __init__(self):
        self.x = None
        self.P = 1000.0
        self.Q = 1.0  # 过程噪声
        self.R = 10.0  # 观测噪声
    
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return measurement
        
        # 预测
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 更新
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x

def main():
    # 摄像头设置
    # camera_index = 1
    camera_index = "./captured_videos/pendulum_20250719_184955.mp4"  # 使用视频文件进行测试

    if type(camera_index) is str:
        # 如果是视频文件路径，使用cv2.VideoCapture读取视频
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        exit()
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # 获取实际参数
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"摄像头设置：{width}x{height} @ {actual_fps}fps")
    
    # 创建追踪器
    tracker = DoublePendulumTracker()
    
    # FPS计算
    frame_times = []
    
    # 数据记录
    data_log = []
    
    print("按键说明：")
    print("q - 退出程序")
    print("r - 重置轨迹")
    print("c - 校准支点（点击鼠标确定支点位置）")
    print("s - 保存数据")
    
    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            tracker.pivot_point = (x, y)
            print(f"支点设置为: ({x}, {y})")
    
    cv2.namedWindow('Double Pendulum Tracker')
    cv2.setMouseCallback('Double Pendulum Tracker', mouse_callback)
    
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取视频帧")
            break
        
        # 计算FPS
        frame_times.append(current_time)
        frame_times = [t for t in frame_times if current_time - t < 1.0]
        fps = len(frame_times) if len(frame_times) > 1 else 0
        
        # 检测红球和绿球
        red_pos, red_contour, red_area = tracker.detect_ball(frame, "red")
        green_pos, green_contour, green_area = tracker.detect_ball(frame, "green")
        
        # 卡尔曼滤波平滑位置
        if red_pos:
            red_pos = (int(tracker.red_filter.update(red_pos[0])), 
                      int(tracker.red_filter.update(red_pos[1])))
        if green_pos:
            green_pos = (int(tracker.green_filter.update(green_pos[0])), 
                        int(tracker.green_filter.update(green_pos[1])))
        
        # 更新轨迹
        if red_pos:
            tracker.red_trail.append(red_pos)
        if green_pos:
            tracker.green_trail.append(green_pos)
        
        # 估计支点和计算角度
        pivot = tracker.estimate_pivot_point(red_pos, green_pos)
        theta1, theta2, L1, L2 = tracker.calculate_angles(pivot, red_pos, green_pos)
        
        # 记录数据
        if theta1 is not None and theta2 is not None:
            tracker.theta1_history.append(theta1)
            tracker.theta2_history.append(theta2)
            tracker.time_history.append(current_time)
            
            # 数据日志
            data_log.append({
                'time': current_time,
                'red_pos': red_pos,
                'green_pos': green_pos,
                'theta1': theta1,
                'theta2': theta2,
                'L1': L1,
                'L2': L2
            })
        
        # 绘制检测结果
        if red_pos:
            cv2.circle(frame, red_pos, 15, (0, 0, 255), 3)
            cv2.putText(frame, "RED", (red_pos[0]-20, red_pos[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if green_pos:
            cv2.circle(frame, green_pos, 15, (0, 255, 0), 3)
            cv2.putText(frame, "GREEN", (green_pos[0]-25, green_pos[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制摆的结构和轨迹
        tracker.draw_pendulum(frame, pivot, red_pos, green_pos)
        tracker.draw_trails(frame)
        
        # 绘制信息
        tracker.draw_info(frame, red_pos, green_pos, theta1, theta2, L1, L2, fps)
        
        # 显示画面
        cv2.imshow('Double Pendulum Tracker', frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.red_trail.clear()
            tracker.green_trail.clear()
            print("轨迹已重置")
        elif key == ord('c'):
            print("点击鼠标左键设置支点位置")
        elif key == ord('s'):
            # 保存数据
            if data_log:
                import json
                filename = f"pendulum_data_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(data_log, f, default=str)
                print(f"数据已保存到 {filename}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 绘制角度时间图（如果有数据）
    if len(tracker.theta1_history) > 10:
        plt.figure(figsize=(12, 8))
        
        times = list(tracker.time_history)
        theta1s = [math.degrees(t) for t in tracker.theta1_history]
        theta2s = [math.degrees(t) for t in tracker.theta2_history]
        
        plt.subplot(2, 1, 1)
        plt.plot(times, theta1s, 'r-', label='Theta1 (红球角度)')
        plt.ylabel('角度 (度)')
        plt.title('双球摆角度变化')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(times, theta2s, 'g-', label='Theta2 (绿球相对角度)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('角度 (度)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    print(f"程序结束，最终FPS: {fps:.1f}")
    print(f"记录了 {len(data_log)} 个数据点")

if __name__ == "__main__":
    main()