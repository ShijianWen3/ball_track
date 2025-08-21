


from ultralytics import YOLO

# 定义数据集路径和配置文件
data_path = './dataset_ball_v8/dataset_ball_v8.yaml'  # 你需要把这个路径指向你的 dataset.yaml 文件
weights_path = 'yolov8s.pt'  # 可以选择预训练模型，yolov8s.pt 是较小的模型，适用于快速实验

# 初始化 YOLO 模型
model = YOLO(weights_path)

# 开始训练
model.train(
    data=data_path,           # 指定数据集配置文件路径
    epochs=100,               # 训练轮次
    batch=10,                 # 批次大小
    imgsz=[640,480],          # 输入图像的尺寸
    device='cpu',             # 使用第一个 GPU，设置为 'cpu' 则在 CPU 上训练
    save_period=10,           # 每 10 轮保存一次模型
    workers=10,                # 使用的线程数
    project='./model_ballv1.0/',   # 输出目录
    name='red1.1',      # 输出模型的名称
    exist_ok=True,            # 如果已有文件夹存在，覆盖文件夹
    patience = 35             #早停功能，若连续patience轮训练模型性能没有提升则停止
)

# 训练完成后会自动保存模型
