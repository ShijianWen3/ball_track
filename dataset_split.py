import random
import shutil
import os

def split_dataset(image_dir, label_dir, output_dir, ratios=(0.7, 0.2, 0.1)):
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)
    
    # 获取所有图片文件
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)
    
    # 计算划分点
    total = len(images)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    
    # 划分并复制文件
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    for split, files in splits.items():
        for img_file in files:
            # 复制图片
            shutil.copy(f'{image_dir}/{img_file}', 
                       f'{output_dir}/images/{split}/{img_file}')
            
            # 复制对应的标签文件
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(f'{label_dir}/{label_file}'):
                shutil.copy(f'{label_dir}/{label_file}', 
                           f'{output_dir}/labels/{split}/{label_file}')

# 使用示例
split_dataset('./dataset/front', './dataset/label', './dataset_v8')