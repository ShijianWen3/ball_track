import cv2
import os
import glob
import re

def get_max_image_number(folder_path):
    """
    检索指定路径下所有.png文件名中的最大数字
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return 0
    
    # 获取所有.png文件
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    
    if not png_files:
        return 0
    
    max_number = 0
    for file_path in png_files:
        # 提取文件名（不包含路径和扩展名）
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # 检查文件名是否全是数字
        if filename_without_ext.isdigit():
            number = int(filename_without_ext)
            max_number = max(max_number, number)
    
    return max_number

def main():
    # 设置保存路径（请修改为您想要的路径）
    save_path = "./dataset"  # 您可以修改这个路径
    
    # 初始化摄像头
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 获取当前最大图像编号
    current_max_number = get_max_image_number(save_path)
    next_number = current_max_number + 1
    
    print(f"数据集采集程序已启动")
    print(f"保存路径: {os.path.abspath(save_path)}")
    print(f"当前最大图像编号: {current_max_number}")
    print(f"下一张图像将保存为: {next_number}.png")
    print("\n操作说明:")
    print("- 按 SPACE 键保存当前帧")
    print("- 按 'q' 键退出程序")
    print("- 按 'r' 键重新检索最大编号")
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # 在图像上显示信息
        info_text = f"Next: {next_number}.png | Press SPACE to save, 'q' to quit"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow('Dataset Collector', frame)
        
        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # 退出程序
            print("程序退出")
            break
        elif key == ord(' '):  # 空格键
            # 保存当前帧
            filename = f"{next_number}.png"
            filepath = os.path.join(save_path, filename)
            
            success = cv2.imwrite(filepath, frame)
            if success:
                print(f"已保存: {filepath}")
                next_number += 1
            else:
                print(f"保存失败: {filepath}")
        elif key == ord('r'):
            # 重新检索最大编号
            current_max_number = get_max_image_number(save_path)
            next_number = current_max_number + 1
            print(f"重新检索完成，下一张图像编号: {next_number}")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()