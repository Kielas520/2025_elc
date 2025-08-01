import cv2
import numpy as np
import math
import time
from model.detector import Detector
from model.tracker import Tracker

def nothing(x):
    pass

def init_board():
    """初始化 OpenCV 窗口和滑动条"""
    cv2.namedWindow('Result', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('Mask', cv2.WINDOW_FREERATIO)
    cv2.moveWindow('Mask', 360, 180)
    cv2.moveWindow('Result', 540, 180)
    cv2.resizeWindow('Mask', 170, 150)
    cv2.resizeWindow('Result', 170, 150)

    cv2.namedWindow('Controls', cv2.WINDOW_FREERATIO)
    cv2.moveWindow('Controls', 0, 0)
    cv2.resizeWindow('Controls', 320, 400)
    cv2.createTrackbar('H Min', 'Controls', 133, 179, nothing)
    cv2.createTrackbar('H Max', 'Controls', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Controls', 6, 255, nothing)
    cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('board_min_area', 'Controls', 18310, 307200, nothing)
    cv2.createTrackbar('board_max_area', 'Controls', 50000, 307200, nothing)
    cv2.createTrackbar('diameter_ratio', 'Controls', 40, 70, nothing)
    cv2.createTrackbar('cx_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('cy_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('show', 'Controls', 0, 1, nothing)
    cv2.createTrackbar('task', 'Controls', 0, 1, nothing)  # 切换任务（0 或 1）

def update_hsv(detector):
    """更新 HSV 参数和检测器设置"""
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Controls')
    board_max_area = cv2.getTrackbarPos('board_max_area', 'Controls')
    diameter_ratio = cv2.getTrackbarPos('diameter_ratio', 'Controls')
    cx_offset = cv2.getTrackbarPos('cx_offset', 'Controls')
    cy_offset = cv2.getTrackbarPos('cy_offset', 'Controls')
    show = cv2.getTrackbarPos('show', 'Controls')
    task = cv2.getTrackbarPos('task', 'Controls')

    detector.board_lower = (h_min, s_min, v_min)
    detector.board_upper = (h_max, s_max, v_max)
    detector.diameter_ratio = diameter_ratio / 100
    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    detector.cx_offset = cx_offset - 30
    detector.cy_offset = cy_offset - 30
    detector.show_img = show
    detector.task = task

def main():
    # 初始化相机
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
        print("Camera open failed")
        return

    # 初始化检测器和跟踪器
    detector = Detector(
        board_color=[(13, 255, 152), (0, 51, 110)],
        board_min_area=18310,
        board_max_area=50000,
        diameter_ratio=0.5
    )
    tracker = Tracker(
        img_width=640,
        img_height=480,
        vfov=100,
        use_kf=False,
        frame_add=10,
        shoot_tol=5,
        ref_point=(-0.04, 0, 0)
    )

    # 初始化窗口
    init_board()

    # 初始化时间变量
    last_time = time.time()
    frame_count = 0
    fps = 0
    dt = 1/30

    try:
        while True:
            # 读取帧
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame")
                break

            # 更新 HSV 和检测参数
            update_hsv(detector)

            # 根据任务执行检测
            if detector.task == 0:
                position = detector.task1(frame)
            elif detector.task == 1:
                position = detector.task2(frame)

            # 跟踪并计算角度
            relative_yaw, relative_pitch = tracker.track(position, dt)
            arrived = tracker.shoot  # 从 tracker 中获取 arrived（shoot 表示是否接近中心）

            # 打印结果
            if relative_yaw is not None and relative_pitch is not None:
                print(f"Relative Pitch: {relative_pitch:.2f}°, Relative Yaw: {relative_yaw:.2f}°, Arrived: {arrived}")
            else:
                print("No target detected")

            # 显示结果
            if detector.show_img == 1:
                result = detector.display(frame)
                cv2.imshow('Mask', detector.board_mask)
                cv2.imshow('Result', result)

            # 计算 FPS
            current_time = time.time()
            frame_count += 1
            dt = current_time - last_time
            elapsed_time = current_time - last_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                last_time = current_time
                print(f"FPS: {fps:.2f}")

            # 按 'q' 退出
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Main loop error: {str(e)}")

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()