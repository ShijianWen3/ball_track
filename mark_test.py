import cv2
import numpy as np

# 初始化摄像头
cap = cv2.VideoCapture(1)

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
        # 绘制左上到右下的绿色虚线对角线
        num_dashes = 30
        for i in range(num_dashes):
            start_frac = i / num_dashes
            end_frac = (i + 0.5) / num_dashes
            if end_frac > 1:
                end_frac = 1
            x1 = int(frame_width * start_frac)
            y1 = int(frame_height * start_frac)
            x2 = int(frame_width * end_frac)
            y2 = int(frame_height * end_frac)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色虚线
        # 绘制左下到右上的绿色虚线对角线
        for i in range(num_dashes):
            start_frac = i / num_dashes
            end_frac = (i + 0.5) / num_dashes
            if end_frac > 1:
                end_frac = 1
            x1 = int(frame_width * start_frac)
            y1 = int(frame_height * (1 - start_frac))
            x2 = int(frame_width * end_frac)
            y2 = int(frame_height * (1 - end_frac))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色虚线

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
