import cv2
import numpy as np

# 回调函数（空函数，用于滑动条）
def nothing(x):
    pass

# 打开摄像头
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口和滑动条
cv2.namedWindow('Control Panel')
cv2.createTrackbar('Bin Min', 'Control Panel', 50, 255, nothing)  # 二值化最小阈值
cv2.createTrackbar('Bin Max', 'Control Panel', 200, 255, nothing) # 二值化最大阈值
cv2.createTrackbar('Invert', 'Control Panel', 1, 1, nothing)      # 二值化反转开关
cv2.createTrackbar('Mode', 'Control Panel', 1, 2, nothing)        # 模式选择: 0角点, 1圆形, 2轮廓

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 获取滑动条参数
    bin_min = cv2.getTrackbarPos('Bin Min', 'Control Panel')
    bin_max = cv2.getTrackbarPos('Bin Max', 'Control Panel')
    invert = cv2.getTrackbarPos('Invert', 'Control Panel')
    mode = cv2.getTrackbarPos('Mode', 'Control Panel')
    
    # 固定阈值二值化
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(blurred, bin_min, bin_max, thresh_type)
    
    # 复制原始帧用于绘制结果
    output = frame.copy()
    
    if mode == 0:  # 角点检测
        # 使用Shi-Tomasi角点检测
        corners = cv2.goodFeaturesToTrack(blurred, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = np.intp(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
    
    elif mode == 1:  # 圆形检测
        # 霍夫圆检测
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
    
    elif mode == 2:  # 轮廓检测
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # 计算轮廓面积，过滤小轮廓
            if cv2.contourArea(contour) > 100:
                # 拟合多边形
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
    
    # 显示原始帧和处理结果
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Detection Result', output)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()