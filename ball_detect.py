import cv2
import numpy as np
from collections import deque


class StableBallTracker:
    def __init__(self):
        # 位置历史记录，用于平滑滤波
        self.position_history = deque(maxlen=5)  # 保存最近5帧的位置
        self.radius_history = deque(maxlen=5)    # 保存最近5帧的半径
        
        # 上一帧的最佳圆心位置，用于连续性检查
        self.last_center = None
        self.last_radius = None
        
        # 卡尔曼滤波器参数（简化版）
        self.kalman_x = cv2.KalmanFilter(4, 2)
        self.kalman_y = cv2.KalmanFilter(4, 2)
        self.init_kalman()
        
        # 检测参数
        self.detection_confidence = 0
        self.min_confidence = 3  # 需要连续检测到3帧才认为稳定
        
    def init_kalman(self):
        """初始化卡尔曼滤波器"""
        # 状态转移矩阵 [x, vx, ax, jx] 或 [y, vy, ay, jy]
        self.kalman_x.transitionMatrix = np.array([[1, 1, 0.5, 0.16],
                                                   [0, 1, 1, 0.5],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 0, 1]], np.float32)
        
        self.kalman_y.transitionMatrix = self.kalman_x.transitionMatrix.copy()
        
        # 测量矩阵
        self.kalman_x.measurementMatrix = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0]], np.float32)
        self.kalman_y.measurementMatrix = self.kalman_x.measurementMatrix.copy()
        
        # 过程噪声协方差
        self.kalman_x.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        self.kalman_y.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        
        # 测量噪声协方差
        self.kalman_x.measurementNoiseCov = 5 * np.eye(2, dtype=np.float32)
        self.kalman_y.measurementNoiseCov = 5 * np.eye(2, dtype=np.float32)
        
        # 后验误差协方差
        self.kalman_x.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman_y.errorCovPost = np.eye(4, dtype=np.float32)
        
    def smooth_position(self, center, radius):
        """使用历史记录平滑位置"""
        if len(self.position_history) == 0:
            return center, radius
            
        # 加权平均，最新的权重更高
        weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(self.position_history)]
        weights = weights / weights.sum()
        
        # 计算加权平均位置
        avg_x = sum(pos[0] * w for pos, w in zip(self.position_history, weights))
        avg_y = sum(pos[1] * w for pos, w in zip(self.position_history, weights))
        avg_radius = sum(r * w for r, w in zip(self.radius_history, weights))
        
        # 与当前检测结果融合
        smooth_x = int(0.7 * center[0] + 0.3 * avg_x)
        smooth_y = int(0.7 * center[1] + 0.3 * avg_y)
        smooth_radius = int(0.7 * radius + 0.3 * avg_radius)
        
        return (smooth_x, smooth_y), smooth_radius
    
    def is_valid_circle(self, center, radius, frame_shape):
        """验证检测到的圆是否合理"""
        x, y = center
        h, w = frame_shape[:2]
        
        # 检查圆心是否在合理范围内
        if x < radius or x > w - radius or y < radius or y > h - radius:
            return False
            
        # 检查半径是否合理
        if radius < 10 or radius > min(w, h) // 3:
            return False
            
        # 如果有历史位置，检查位置变化是否合理
        if self.last_center is not None:
            distance = np.sqrt((x - self.last_center[0])**2 + (y - self.last_center[1])**2)
            # 位置变化不应该太大（根据实际情况调整）
            if distance > 100:  # 像素
                return False
                
        return True
    
    def detect_best_circle(self, processed_frame, original_frame):
        """检测最佳圆形，增加稳定性"""
        # 多参数霍夫圆检测
        circles_sets = []
        
        # 参数组合1：严格检测
        circles1 = cv2.HoughCircles(processed_frame, 
                                   cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                   param1=50, param2=20, minRadius=15, maxRadius=80)
        if circles1 is not None:
            circles_sets.append(circles1[0])
            
        # 参数组合2：宽松检测
        circles2 = cv2.HoughCircles(processed_frame, 
                                   cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                                   param1=30, param2=15, minRadius=10, maxRadius=100)
        if circles2 is not None:
            circles_sets.append(circles2[0])
        
        # 合并所有检测结果
        all_circles = []
        for circles in circles_sets:
            all_circles.extend(circles)
            
        if not all_circles:
            return None, None
            
        # 筛选有效圆
        valid_circles = []
        for circle in all_circles:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            if self.is_valid_circle((x, y), r, original_frame.shape):
                valid_circles.append((x, y, r))
        
        if not valid_circles:
            return None, None
            
        # 如果有历史位置，优先选择距离最近的圆
        if self.last_center is not None:
            distances = []
            for x, y, r in valid_circles:
                dist = np.sqrt((x - self.last_center[0])**2 + (y - self.last_center[1])**2)
                distances.append(dist)
            best_idx = np.argmin(distances)
            best_circle = valid_circles[best_idx]
        else:
            # 否则选择半径最大的圆
            best_circle = max(valid_circles, key=lambda c: c[2])
            
        return (best_circle[0], best_circle[1]), best_circle[2]


def adaptive_threshold_denoise(img):
    """自适应阈值去噪"""
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def otsu_threshold_denoise(img):
    """OTSU自动阈值去噪"""
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th


def preprocess_frame(frame):
    """预处理帧，提取绿色区域"""
    # 转换到HSV色域
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 绿色范围 - 可以根据实际情况调整
    lower_green = np.array([35, 50, 20])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(mask_green, (9, 9), 2)
    
    # OTSU阈值
    denoised = otsu_threshold_denoise(blurred)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 额外的腐蚀操作，减少噪声
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final = cv2.morphologyEx(closed, cv2.MORPH_ERODE, kernel_small)
    
    return final, mask_green


def main():
    # 初始化跟踪器
    tracker = StableBallTracker()
    
    # 加载视频
    video_path = './captured_videos/test1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 预处理帧
        processed_frame, mask_green = preprocess_frame(frame)
        
        # 检测圆
        center, radius = tracker.detect_best_circle(processed_frame, frame)
        
        if center is not None and radius is not None:
            # 更新历史记录
            tracker.position_history.append(center)
            tracker.radius_history.append(radius)
            
            # 平滑位置
            smooth_center, smooth_radius = tracker.smooth_position(center, radius)
            
            # 更新置信度
            tracker.detection_confidence = min(tracker.detection_confidence + 1, tracker.min_confidence)
            
            # 只有置信度足够高时才显示结果
            if tracker.detection_confidence >= tracker.min_confidence:
                # 绘制圆
                cv2.circle(frame, smooth_center, smooth_radius, (0, 255, 0), 2)
                cv2.circle(frame, smooth_center, 2, (0, 0, 255), 3)
                
                # 显示信息
                cv2.putText(frame, f'Center: {smooth_center}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Radius: {smooth_radius}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 更新上一帧位置
            tracker.last_center = smooth_center
            tracker.last_radius = smooth_radius
            
        else:
            # 减少置信度
            tracker.detection_confidence = max(tracker.detection_confidence - 1, 0)
        
        # 显示结果
        cv2.imshow('Ball Detection', frame)
        cv2.imshow('Processed', processed_frame)
        cv2.imshow('Green Mask', mask_green)
        
        # 按ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC键或q键
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()