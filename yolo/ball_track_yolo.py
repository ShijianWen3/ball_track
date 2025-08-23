import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import argparse
import torch
import os
import threading
class RGBBallTracker:
    def __init__(self, model_path, confidence_threshold=0.55, device='auto'):
        """
        初始化RGB球追踪器
        
        Args:
            model_path: YOLO模型文件路径
            confidence_threshold: 置信度阈值
            device: 设备选择 ('auto', 'cpu', 'cuda:0', 'cuda:1', etc.)
        """
        # 自动选择最佳设备
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"✓ 自动选择GPU: {torch.cuda.get_device_name(0)}")
                print(f"✓ GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device = 'cpu'
                print("⚠ GPU不可用，使用CPU")
        else:
            if device.startswith('cuda') and not torch.cuda.is_available():
                print("⚠ 指定GPU不可用，切换到CPU")
                device = 'cpu'
            elif device.startswith('cuda'):
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                if gpu_id < torch.cuda.device_count():
                    print(f"✓ 使用指定GPU: {torch.cuda.get_device_name(gpu_id)}")
                else:
                    print(f"⚠ GPU {gpu_id} 不存在，使用GPU 0")
                    device = 'cuda:0'
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        print(f"📦 正在加载模型...")
        self.model = YOLO(model_path)
        
        # 将模型移动到指定设备
        if device != 'cpu':
            print(f"🔄 将模型移动到 {device}...")
            self.model.to(device)
        
        print(f"✓ 模型已加载到 {device}")
        
        # GPU优化设置
        if device.startswith('cuda'):
            self._setup_gpu_optimization()
        
        # 定义颜色映射 (BGR格式)
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0)
        }
        
        # 类别名称映射（根据你的模型训练时的类别顺序调整）
        self.class_names = {
            0: 'red_ball',
            1: 'green_ball', 
            2: 'blue_ball'
        }
        
        # 追踪历史记录
        self.tracking_history = defaultdict(list)
        self.frame_count = 0
        
        # 性能监控
        self.inference_times = []
        self.total_inference_time = 0
        
        # GPU预热
        if device.startswith('cuda'):
            self._warmup_model()
    
    def _setup_gpu_optimization(self):
        """设置GPU优化"""
        try:
            # 启用CUDNN基准模式（固定输入尺寸时有效）
            torch.backends.cudnn.benchmark = True
            print("✓ CUDNN基准模式已启用")
        except Exception as e:
            print(f"⚠ GPU优化设置失败: {e}")
    
    def _warmup_model(self, warmup_frames=5):
        """GPU模型预热"""
        print("🔥 GPU模型预热中...")
        dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        for i in range(warmup_frames):
            with torch.no_grad():
                _ = self.model(dummy_frame, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        print("✓ GPU预热完成")
        
    def process_frame(self, frame, isPrintInfo=True):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            isPrintInfo: 是否显示统计信息
            
        Returns:
            processed_frame: 处理后的图像帧
            detections: 检测结果
        """


        # 记录推理开始时间
        inference_start = time.perf_counter()
        
        # 运行YOLO推理 - GPU优化推理
        with torch.no_grad():  # 禁用梯度计算节省显存
            if self.device.startswith('cuda'):
                # GPU推理时使用混合精度加速
                with torch.cuda.amp.autocast():
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                # 确保GPU操作完成
                torch.cuda.synchronize()
            else:
                # CPU推理
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # 记录推理时间
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        self.inference_times.append(inference_time)
        self.total_inference_time += inference_time
        
        # 复制帧用于绘制
        processed_frame = frame.copy()
        detections = []
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    # 获取类别名称
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 记录检测结果
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    }
                    detections.append(detection)

        # 只保留每种球类型置信度最高的一个
        best_detections = {}
        for det in detections:
            cname = det['class_name']
            if cname not in best_detections or det['confidence'] > best_detections[cname]['confidence']:
                best_detections[cname] = det
        detections = list(best_detections.values())

        # 绘制和追踪仅对筛选后的检测进行
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = self.get_color_for_class(class_name)
            # 更新追踪历史
            self.tracking_history[class_name].append((center_x, center_y))
            if len(self.tracking_history[class_name]) > 50:
                self.tracking_history[class_name].pop(0)
            # 绘制边界框
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            # 绘制标签
            label = f'{class_name}: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(processed_frame, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        color, -1)
            cv2.putText(processed_frame, label, 
                      (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # 绘制中心点
            cv2.circle(processed_frame, (center_x, center_y), 5, color, -1)

        # 绘制追踪轨迹
        self.draw_tracking_trails(processed_frame)

        # 绘制统计信息
        if isPrintInfo:
            self.draw_statistics(processed_frame, detections, inference_time)

        return processed_frame, detections
    
    def get_color_for_class(self, class_name):
        """根据类别名称获取颜色"""
        if 'red' in class_name.lower():
            return self.colors['red']
        elif 'green' in class_name.lower():
            return self.colors['green']
        elif 'blue' in class_name.lower():
            return self.colors['blue']
        else:
            return (128, 128, 128)  # 灰色作为默认颜色
    
    def draw_tracking_trails(self, frame):
        """绘制追踪轨迹"""
        for class_name, points in self.tracking_history.items():
            if len(points) > 1:
                color = self.get_color_for_class(class_name)
                # 绘制轨迹线
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 1)
    
    def draw_statistics(self, frame, detections, inference_time):
        """绘制统计信息"""
        h, w = frame.shape[:2]
        
        # 统计各类球的数量
        stats = defaultdict(int)
        for det in detections:
            stats[det['class_name']] += 1
        
        # 计算性能指标
        avg_inference_time = np.mean(self.inference_times[-30:]) if self.inference_times else 0
        theoretical_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 绘制背景 - 扩大以显示更多信息
        info_height = 180 if self.device.startswith('cuda') else 140
        cv2.rectangle(frame, (10, 10), (350, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, info_height), (255, 255, 255), 2)
        
        # 显示基本信息
        y_pos = 30
        cv2.putText(frame, f'Frame: {self.frame_count}', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        y_pos += 20
        cv2.putText(frame, f'Device: {self.device}', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示性能信息
        y_pos += 20
        cv2.putText(frame, f'Inference: {inference_time*1000:.1f}ms', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        y_pos += 20
        cv2.putText(frame, f'Avg FPS: {theoretical_fps:.1f}', 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 显示GPU信息
        if self.device.startswith('cuda') and torch.cuda.is_available():
            y_pos += 20
            memory_used = torch.cuda.memory_allocated() / 1024**3
            cv2.putText(frame, f'GPU Mem: {memory_used:.2f}GB', 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 显示检测统计
        y_pos += 25
        for class_name in ['red_ball', 'green_ball', 'blue_ball']:
            count = stats.get(class_name, 0)
            color = self.get_color_for_class(class_name)
            cv2.putText(frame, f'{class_name}: {count}', 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_pos += 20
        
        self.frame_count += 1
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.inference_times:
            return None
        
        return {
            'avg_inference_ms': np.mean(self.inference_times) * 1000,
            'min_inference_ms': np.min(self.inference_times) * 1000,
            'max_inference_ms': np.max(self.inference_times) * 1000,
            'theoretical_fps': 1 / np.mean(self.inference_times),
            'total_frames': len(self.inference_times)
        }

def test_on_webcam(model_path, isPrintInfo=True, device='auto', isSaveTraceEnable=False):
    """使用摄像头测试模型"""
    tracker = RGBBallTracker(model_path, device=device)
    cap = cv2.VideoCapture(1)
    
    # 摄像头优化设置
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("开始摄像头测试，按 'q' 退出")
    fps_counter = 0
    start_time = time.time()

    if isSaveTraceEnable:
        # 轨迹记录相关
        tracking = False
        ball_tracks = {'red_ball': [], 'green_ball': [], 'blue_ball': []}
        print("按 s 开始/暂停追踪，按 q 退出并保存轨迹")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame, detections = tracker.process_frame(frame, isPrintInfo)
            
            # 计算实际FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = fps_counter / elapsed
                print(f"实际FPS: {actual_fps:.2f}, 检测到球数量: {len(detections)}")
                
                # 显示性能统计
                stats = tracker.get_performance_stats()
                if stats:
                    print(f"推理时间: {stats['avg_inference_ms']:.1f}ms, "
                          f"理论FPS: {stats['theoretical_fps']:.1f}")
            
            # 显示结果
            cv2.imshow('RGB Ball Tracking - GPU Accelerated', processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if isSaveTraceEnable:
                if tracking:
                    update_ball_tracks(ball_tracks, detections)

            # 按 'q' 退出
            
            if key == ord('q'):
                
                break
            elif key == ord('s'):
                if isSaveTraceEnable:
                    tracking = not tracking
                    print("追踪" + ("开始" if tracking else "暂停"))
                else:
                    pass
        if isSaveTraceEnable:
            save_ball_tracks(ball_tracks)
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 最终性能报告
        stats = tracker.get_performance_stats()
        if stats:
            print(f"\n=== 最终性能报告 ===")
            print(f"设备: {tracker.device}")
            print(f"总帧数: {stats['total_frames']}")
            print(f"平均推理时间: {stats['avg_inference_ms']:.2f}ms")
            print(f"理论最大FPS: {stats['theoretical_fps']:.1f}")
            print(f"推理时间范围: {stats['min_inference_ms']:.1f}-{stats['max_inference_ms']:.1f}ms")

def test_on_video(model_path, video_path, output_path=None, device='auto'):
    """在视频文件上测试模型"""
    tracker = RGBBallTracker(model_path, device=device)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    print(f"使用设备: {tracker.device}")
    
    # 设置输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed_frame, detections = tracker.process_frame(frame, True)
            
            # 保存到输出视频
            if output_path:
                out.write(processed_frame)
            
            # 显示进度
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                processing_fps = frame_idx / elapsed
                print(f"处理进度: {progress:.1f}% ({frame_idx}/{total_frames}), "
                      f"处理速度: {processing_fps:.1f}fps")
            
            # 显示结果（可选）
            cv2.imshow('RGB Ball Tracking', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n用户中断处理")
    
    finally:
        cap.release()
        if output_path:
            out.release()
            print(f"输出视频已保存到: {output_path}")
        
        cv2.destroyAllWindows()
        
        # 最终性能报告
        elapsed_total = time.time() - start_time
        avg_processing_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0
        
        stats = tracker.get_performance_stats()
        if stats:
            print(f"\n=== 视频处理性能报告 ===")
            print(f"设备: {tracker.device}")
            print(f"处理帧数: {frame_idx}/{total_frames}")
            print(f"总处理时间: {elapsed_total:.1f}s")
            print(f"平均处理速度: {avg_processing_fps:.1f}fps")
            print(f"平均推理时间: {stats['avg_inference_ms']:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description='RGB球追踪YOLO模型测试 - GPU加速版')
    parser.add_argument('--model', required=True, help='YOLO模型文件路径')
    parser.add_argument('--source', default='webcam', 
                       help='输入源: webcam 或视频文件路径')
    parser.add_argument('--output', help='输出视频路径（仅对视频文件有效）')
    parser.add_argument('--conf', type=float, default=0.55, 
                       help='置信度阈值')
    parser.add_argument('--device', default='auto',
                       help='推理设备: auto, cpu, cuda:0, cuda:1 等')
    parser.add_argument('--info', action='store_true',
                       help='显示统计信息')
    
    args = parser.parse_args()
    
    print(f"🚀 RGB球追踪模型测试 - GPU加速版")
    print(f"📦 模型文件: {args.model}")
    print(f"🖥️ 推理设备: {args.device}")
    print(f"🎯 置信度阈值: {args.conf}")
    print(f"📊 显示信息: {args.info}")
    
    if args.source == 'webcam':
        test_on_webcam(args.model, args.info, args.device)
    else:
        test_on_video(args.model, args.source, args.output, args.device)


def save_ball_tracks(tracks_dict, out_dir=".\\traces",suffix:int = 1):
    """
    保存球轨迹到txt文件，tracks_dict: {class_name: [(x, y), ...]}
    """
    for cname, points in tracks_dict.items():
        fname = os.path.join(out_dir, f"{cname}_{suffix}.txt")
        with open(fname, "w") as f:
            for x, y in points:
                f.write(f"{x},{y}\n")

def update_ball_tracks(tracks_dict, detections):
    """
    根据当前帧检测结果，更新轨迹字典
    """
    for det in detections:
        cname = det['class_name']
        center = det['center']
        tracks_dict[cname].append(center)




if __name__ == "__main__":
    # 如果直接运行，使用默认参数进行测试
    model_path = "./yolo/ball.pt"
    
    print("🚀 RGB球追踪模型测试 - GPU加速版")
    print("=" * 50)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠ 未检测到GPU，将使用CPU")
    
    print("=" * 50)
    print("1. 摄像头测试")
    print("2. 视频文件测试")
    
    choice = input("请选择测试方式 (1/2): ")

    isPrintInfo = input('是否绘制统计信息？(y/n): ')
    if isPrintInfo.lower() == 'y':
        print("✓ 将绘制统计信息")
        isPrintInfo = True
    else:
        print("✗ 不绘制统计信息")
        isPrintInfo = False
    
    device = input('选择设备 (auto/cpu/cuda:0): ').strip()
    if not device:
        device = 'auto'
    
    print(f"📱 选择设备: {device}")
    
    if choice == "1":
        test_on_webcam(model_path, isPrintInfo, device, True)
    elif choice == "2":
        video_path = input("请输入视频文件路径: ")
        output_path = input("请输入输出视频路径（回车跳过）: ")
        if not output_path:
            output_path = None
        test_on_video(model_path, video_path, output_path, device)
    else:
        print("❌ 无效选择")