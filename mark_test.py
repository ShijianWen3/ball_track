import cv2
import numpy as np

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 设置窗口大小
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
isDisplay = True

# 调试主循环
while True:
    # 从摄像头获取一帧图像
    ret, frame = cap.read()
    if not ret:
        print("无法从摄像头读取图像")
        break

    if isDisplay:
        # 绘制水平线（红色）
        cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 0, 255), 1)  # 红色水平线

        # 绘制中垂线（蓝色）
        cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 0, 0), 1)  # 蓝色中垂线

        # 绘制对角线（绿色）
        cv2.line(frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 1)  # 绿色左上到右下对角线
        cv2.line(frame, (0, frame_height), (frame_width, 0), (0, 255, 0), 1)  # 绿色左下到右上对角线

    # 显示图像
    cv2.imshow('Camera Calibration', frame)

    # 键盘输入控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按'q'退出
        break
    elif key == ord('d'):  # 按'd'切换显示状态
        isDisplay = not isDisplay
# 释放摄像头和窗口
cap.release()
cv2.destroyAllWindows()
