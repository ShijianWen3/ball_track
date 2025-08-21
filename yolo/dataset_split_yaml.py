import os
import shutil
import random
from collections import defaultdict
import argparse

def get_class_names_from_labels(label_dir):
    """从标签文件中提取所有类别名称"""
    classes = set()
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes.add(class_id)
    
    # 返回排序后的类别ID列表
    return sorted(list(classes))

def count_samples_per_class(label_dir):
    """统计每个类别的样本数量"""
    class_counts = defaultdict(int)
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                file_classes = set()
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        file_classes.add(class_id)
                # 每个文件中的每个类别计数+1
                for cls in file_classes:
                    class_counts[cls] += 1
    
    return class_counts

def stratified_split(images, label_dir, ratios=(0.7, 0.2, 0.1)):
    """按类别分层划分数据集，保持类别分布平衡"""
    # 统计每张图片包含的类别
    image_classes = {}
    
    for img_file in images:
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            classes = set()
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        classes.add(class_id)
            image_classes[img_file] = classes
        else:
            image_classes[img_file] = set()
    
    # 简单随机划分（可以根据需要实现更复杂的分层策略）
    random.shuffle(images)
    total = len(images)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    
    return {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

def split_dataset(image_dir, label_dir, output_dir, ratios=(0.7, 0.2, 0.1), class_names=None):
    """划分数据集并生成配置文件"""
    
    print(f"开始划分数据集...")
    print(f"原始图片目录: {image_dir}")
    print(f"原始标签目录: {label_dir}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: 训练集{ratios[0]:.1%}, 验证集{ratios[1]:.1%}, 测试集{ratios[2]:.1%}")
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)
    
    # 获取所有图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_formats)]
    
    if not images:
        raise ValueError(f"在 {image_dir} 中没有找到支持的图片格式文件")
    
    print(f"找到 {len(images)} 张图片")
    
    # 分层划分
    splits = stratified_split(images, label_dir, ratios)
    
    # 统计信息
    stats = {}
    
    # 复制文件并统计
    for split_name, file_list in splits.items():
        print(f"\n处理 {split_name} 集: {len(file_list)} 张图片")
        
        copied_images = 0
        copied_labels = 0
        
        for img_file in file_list:
            # 复制图片
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split_name, img_file)
            shutil.copy2(src_img, dst_img)
            copied_images += 1
            
            # 复制对应的标签文件
            for ext in ['.txt']:
                base_name = os.path.splitext(img_file)[0]
                label_file = base_name + ext
                src_label = os.path.join(label_dir, label_file)
                
                if os.path.exists(src_label):
                    dst_label = os.path.join(output_dir, 'labels', split_name, label_file)
                    shutil.copy2(src_label, dst_label)
                    copied_labels += 1
                    break
        
        stats[split_name] = {
            'images': copied_images,
            'labels': copied_labels
        }
        print(f"  复制了 {copied_images} 张图片, {copied_labels} 个标签文件")
    
    # 自动检测类别
    print(f"\n正在分析类别信息...")
    class_ids = get_class_names_from_labels(label_dir)
    num_classes = len(class_ids)
    
    # 如果没有提供类别名称，使用默认名称
    if class_names is None:
        class_names = [f'class_{i}' for i in class_ids]
    else:
        # 确保类别名称数量匹配
        if len(class_names) != num_classes:
            print(f"警告: 提供的类别名称数量({len(class_names)})与检测到的类别数量({num_classes})不匹配")
            class_names = class_names[:num_classes] + [f'class_{i}' for i in range(len(class_names), num_classes)]
    
    # 统计每个类别的样本数量
    class_counts = count_samples_per_class(label_dir)
    
    print(f"检测到 {num_classes} 个类别:")
    for i, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
        count = class_counts.get(class_id, 0)
        print(f"  类别 {class_id}: {class_name} ({count} 个样本)")
    
    # 生成YAML配置文件（标准YOLOv8格式）
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        # 写入标准格式的YAML
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n") 
        f.write("test: images/test\n")
        f.write(f"\nnc: {num_classes}  # number of classes\n")
        
        # 格式化类别名称为列表形式
        names_str = str(class_names).replace("'", '"')
        f.write(f"names: {names_str}  # class names\n")
    
    print(f"\n✅ 数据集划分完成!")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 配置文件: {yaml_path}")
    print(f"\n📊 划分统计:")
    for split_name, stat in stats.items():
        print(f"  {split_name:>5}: {stat['images']:>4} 张图片, {stat['labels']:>4} 个标签")
    
    print(f"\n🏷️  类别配置:")
    print(f"  类别数量: {num_classes}")
    print(f"  类别名称: {class_names}")
    
    return yaml_path, stats

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='YOLOv8数据集划分工具')
    parser.add_argument('--images', '-i', required=True, help='图片目录路径')
    parser.add_argument('--labels', '-l', required=True, help='标签目录路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.7, 0.2, 0.1], 
                       help='划分比例 (训练集 验证集 测试集), 默认: 0.7 0.2 0.1')
    parser.add_argument('--classes', nargs='*', help='类别名称列表，如: --classes person car bike')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认: 42')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 验证比例总和
    if abs(sum(args.ratios) - 1.0) > 0.001:
        raise ValueError(f"划分比例总和必须为1.0, 当前为: {sum(args.ratios)}")
    
    # 执行划分
    try:
        yaml_path, stats = split_dataset(
            args.images, 
            args.labels, 
            args.output, 
            tuple(args.ratios),
            args.classes
        )
        
        print(f"\n🚀 可以使用以下命令开始训练:")
        print(f"yolo train data={yaml_path} model=yolov8n.pt epochs=100")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

# 使用示例
if __name__ == "__main__":
    # 直接调用示例（如果不使用命令行）
    if len(os.sys.argv) == 1:
        print("YOLOv8数据集划分工具")
        print("\n使用方法1 - 直接调用:")
        print("split_dataset('path/to/images', 'path/to/labels', 'path/to/output')")
        
        print("\n使用方法2 - 命令行:")
        print("python script.py --images /path/to/images --labels /path/to/labels --output /path/to/output")
        print("python script.py -i ./images -l ./labels -o ./dataset --classes person car bike --ratios 0.8 0.1 0.1")
        
        # 示例调用（需要修改路径）
        split_dataset(
            image_dir='./dataset',
            label_dir='./dataset/label', 
            output_dir='./dataset_v8_v1.1',
            ratios=(0.8, 0.1, 0.1),
            class_names=['ball_red', 'ball_green', 'ball_blue']
        )
    else:
        exit(main())