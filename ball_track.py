import cv2
import numpy as np
from collections import deque
import math

class SingleBallPerColorTracker:
    def __init__(self):
        # 颜色定义 (HSV范围) - 根据需要启用颜色
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),      # 红色范围1
                (np.array([160, 100, 100]), np.array([180, 255, 255]))    # 红色范围2
            ],
            'green': [(np.array([50, 50, 20]), np.array([90, 255, 255]))],
            'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        }
        
        # 启用的颜色 - 可以根据需要开关
        # self.enabled_colors = ['red', 'green', 'blue', 'yellow']  # 修改这里来启用/禁用颜色
        self.enabled_colors = ['green']  
        # 每个颜色对应的BGR颜色（用于绘制）
        self.draw_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        
        # 每种颜色只维护一个轨迹
        self.balls = {}  # 格式: {color: {center, radius, confidence, history, disappeared}}
        
        # 检测参数
        self.min_radius = 20
        self.max_radius = 100
        self.min_contour_area = 40
        
        # 跟踪参数
        self.max_disappeared = 5   # 最大消失帧数
        self.history_length = 8    # 轨迹历史长度
        self.confidence_threshold = 0.2  # 最低置信度阈值
        
    def create_color_mask(self, hsv_frame, color_name):
        """为指定颜色创建掩码"""
        masks = []
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_frame, lower, upper)
            masks.append(mask)
        
        # 合并所有掩码（主要用于红色的两个范围）
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def preprocess_mask(self, mask):
        """预处理掩码"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        
        # 形态学操作 - 去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        self.closed = closed
        
        return closed
    
    def find_best_circle(self, mask, color_name):
        """为指定颜色找到最佳的一个圆"""
        candidates = []
        
        # 方法1: 霍夫圆检测
        hough_circles = cv2.HoughCircles(
            mask, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=25, 
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        
        if hough_circles is not None:
            hough_circles = np.uint16(np.around(hough_circles))
            for circle in hough_circles[0, :]:
                x, y, r = circle
                candidates.append({
                    'center': (x, y),
                    'radius': r,
                    'confidence': 0.8,
                    'method': 'hough'
                })
        
        # 方法2: 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # 计算轮廓的最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            if self.min_radius <= radius <= self.max_radius:
                # 计算圆形度
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.4:  # 圆形度阈值
                        candidates.append({
                            'center': center,
                            'radius': radius,
                            'confidence': min(circularity, 0.9),
                            'method': 'contour'
                        })
        
        if not candidates:
            return None
        
        # 选择最佳候选
        best_candidate = None
        
        # 如果该颜色之前有轨迹，优先选择距离最近的
        if color_name in self.balls:
            last_center = self.balls[color_name]['center']
            min_distance = float('inf')
            
            for candidate in candidates:
                distance = math.sqrt(
                    (candidate['center'][0] - last_center[0])**2 + 
                    (candidate['center'][1] - last_center[1])**2
                )
                # 结合距离和置信度
                score = candidate['confidence'] * 0.7 + (1.0 / (1.0 + distance/50)) * 0.3
                candidate['score'] = score
                
                if distance < 80 and score > getattr(best_candidate, 'score', 0):
                    best_candidate = candidate
        
        # 如果没找到合适的，或者是第一次检测，选择置信度最高的
        if best_candidate is None:
            best_candidate = max(candidates, key=lambda x: x['confidence'])
        
        return best_candidate
    
    def update_ball_track(self, color_name, detection):
        """更新指定颜色球的轨迹"""
        if detection is None:
            # 没有检测到，增加消失计数
            if color_name in self.balls:
                self.balls[color_name]['disappeared'] += 1
            return
        
        center = detection['center']
        radius = detection['radius']
        confidence = detection['confidence']
        
        if color_name not in self.balls:
            # 创建新轨迹
            self.balls[color_name] = {
                'center': center,
                'radius': radius,
                'confidence': confidence,
                'history': deque([center], maxlen=self.history_length),
                'disappeared': 0,
                'stable_frames': 1
            }
        else:
            # 更新现有轨迹
            old_center = self.balls[color_name]['center']
            old_radius = self.balls[color_name]['radius']
            
            # 位置平滑
            alpha = 0.7  # 新位置权重
            smooth_center = (
                int(alpha * center[0] + (1-alpha) * old_center[0]),
                int(alpha * center[1] + (1-alpha) * old_center[1])
            )
            smooth_radius = int(alpha * radius + (1-alpha) * old_radius)
            
            self.balls[color_name].update({
                'center': smooth_center,
                'radius': smooth_radius,
                'confidence': max(confidence, self.balls[color_name]['confidence'] * 0.9),
                'disappeared': 0,
                'stable_frames': min(self.balls[color_name]['stable_frames'] + 1, 10)
            })
            
            self.balls[color_name]['history'].append(smooth_center)
    
    def process_frame(self, frame):
        """处理一帧图像"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 处理每种启用的颜色
        for color_name in self.enabled_colors:
            # 创建颜色掩码
            mask = self.create_color_mask(hsv, color_name)
            
            # 预处理
            processed_mask = self.preprocess_mask(mask)
            
            # 找到最佳圆
            detection = self.find_best_circle(processed_mask, color_name)
            
            # 更新轨迹
            self.update_ball_track(color_name, detection)
        
        # 清理长时间消失的轨迹
        colors_to_remove = []
        for color_name in self.balls:
            if self.balls[color_name]['disappeared'] > self.max_disappeared:
                colors_to_remove.append(color_name)
        
        for color_name in colors_to_remove:
            del self.balls[color_name]
    
    def draw_results(self, frame):
        """绘制跟踪结果"""
        for color_name, ball in self.balls.items():
            # 只绘制稳定的球（检测到几帧以上且置信度足够）
            if (ball['stable_frames'] >= 3 and 
                ball['confidence'] >= self.confidence_threshold and
                ball['disappeared'] == 0):
                
                center = ball['center']
                radius = ball['radius']
                confidence = ball['confidence']
                
                # 获取绘制颜色
                draw_color = self.draw_colors[color_name]
                
                # 绘制圆
                cv2.circle(frame, center, radius, draw_color, 2)
                cv2.circle(frame, center, 3, (255, 255, 255), -1)
                
                # 绘制轨迹（如果历史足够长）
                if len(ball['history']) > 1:
                    points = list(ball['history'])
                    for i in range(1, len(points)):
                        alpha = i / len(points)  # 透明度渐变
                        thickness = max(1, int(2 * alpha))
                        cv2.line(frame, points[i-1], points[i], draw_color, thickness)
                
                # 绘制标签
                label = f"{color_name.upper()} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_pos = (center[0] - label_size[0]//2, center[1] - radius - 15)
                
                # 确保标签在图像范围内
                label_pos = (
                    max(0, min(label_pos[0], frame.shape[1] - label_size[0])),
                    max(label_size[1], label_pos[1])
                )
                
                # 绘制标签背景
                cv2.rectangle(frame, 
                             (label_pos[0] - 3, label_pos[1] - label_size[1] - 3),
                             (label_pos[0] + label_size[0] + 3, label_pos[1] + 3),
                             draw_color, -1)
                
                # 绘制标签文字
                cv2.putText(frame, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示统计信息
        active_balls = sum(1 for ball in self.balls.values() 
                          if ball['disappeared'] == 0 and ball['stable_frames'] >= 3)
        stats_text = f"Active Balls: {active_balls}"
        cv2.putText(frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


def main():
    # 初始化跟踪器
    tracker = SingleBallPerColorTracker()
    
    # 如果只想跟踪某些颜色，可以修改这里
    tracker.enabled_colors = ['green']  # 例如只跟踪绿色球
    
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
        
        # 处理帧
        tracker.process_frame(frame)
        
        # 绘制结果
        result_frame = tracker.draw_results(frame.copy())
        
        # 显示帧信息
        cv2.putText(result_frame, f'Frame: {frame_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示当前跟踪的颜色
        enabled_text = f"Colors: {', '.join(tracker.enabled_colors)}"
        cv2.putText(result_frame, enabled_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示结果
        cv2.imshow('Single Ball Per Color Tracking', result_frame)
        cv2.imshow('Mask', tracker.closed)
        
        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC键或q键退出
            break
        elif key == ord('1'):  # 按1键切换红色跟踪
            if 'red' in tracker.enabled_colors:
                tracker.enabled_colors.remove('red')
                if 'red' in tracker.balls:
                    del tracker.balls['red']
            else:
                tracker.enabled_colors.append('red')
        elif key == ord('2'):  # 按2键切换绿色跟踪
            if 'green' in tracker.enabled_colors:
                tracker.enabled_colors.remove('green')
                if 'green' in tracker.balls:
                    del tracker.balls['green']
            else:
                tracker.enabled_colors.append('green')
        elif key == ord('3'):  # 按3键切换蓝色跟踪
            if 'blue' in tracker.enabled_colors:
                tracker.enabled_colors.remove('blue')
                if 'blue' in tracker.balls:
                    del tracker.balls['blue']
            else:
                tracker.enabled_colors.append('blue')
        elif key == ord('4'):  # 按4键切换黄色跟踪
            if 'yellow' in tracker.enabled_colors:
                tracker.enabled_colors.remove('yellow')
                if 'yellow' in tracker.balls:
                    del tracker.balls['yellow']
            else:
                tracker.enabled_colors.append('yellow')
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()