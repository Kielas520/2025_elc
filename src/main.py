import cv2
import numpy as np
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import model.stepper as Stepper
import multiprocessing
import time
from collections import deque

def nothing(x):
    pass

def init_board():
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
    cv2.createTrackbar('yaw_pid', 'Controls', 40, 300, nothing)
    cv2.createTrackbar('pitch_pid', 'Controls', 40, 300, nothing)
    cv2.createTrackbar('yaw_tol', 'Controls', 1, 10, nothing)
    cv2.createTrackbar('pitch_tol', 'Controls', 1, 10, nothing)

def update_hsv():
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Controls')
    board_max_area = cv2.getTrackbarPos('board_max_area', 'Controls')
    diameter_ratio = cv2.getTrackbarPos('diameter_ratio', 'Controls')
    yaw_pid = cv2.getTrackbarPos('yaw_pid', 'Controls')
    pitch_pid = cv2.getTrackbarPos('pitch_pid', 'Controls')
    yaw_tol = cv2.getTrackbarPos('yaw_tol', 'Controls')
    pitch_tol = cv2.getTrackbarPos('pitch_tol', 'Controls')

    detector.board_lower = (h_min, s_min, v_min)
    detector.board_upper = (h_max, s_max, v_max)
    detector.diameter_ratio = diameter_ratio / 100
    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    
    if yaw_pid != 0:
        tracker.yaw_pid = yaw_pid / 100
    if pitch_pid != 0:
        tracker.pitch_pid = pitch_pid / 100

    tracker.yaw_tol = yaw_tol
    tracker.pitch_tol = pitch_tol
    

def tracking_and_steering_thread(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running):
    """跟踪和转向线程
    :param tracker: Tracker 实例
    :param stepper_yaw: Stepper.MotorController 实例（yaw）
    :param stepper_pitch: Stepper.MotorController 实例（pitch）
    :param position_queue: 共享队列，接收 detector 的 position 数据
    :param dt_queue: 共享队列，接收主进程的 dt 数据
    :param running: 运行标志
    """
    yaw_history = deque(maxlen=4)  # 存储前四帧的 yaw 值
    pitch_history = deque(maxlen=4)  # 存储前四帧的 pitch 值（仅为记录，实际不使用）

    while running.is_set():
        dt = 1/30  # 默认值，防止队列为空
        if not dt_queue.empty():
            dt = dt_queue.get()  # 获取最新的 dt
        if not position_queue.empty():
            position = position_queue.get()
            yaw, pitch = tracker.track(position, dt)
            
            # 存储 yaw 和 pitch 值
            yaw_history.append(yaw)
            pitch_history.append(pitch)

            if yaw is None or pitch is None:
                yaw = 0
                pitch = 0
                # 当 yaw 或 pitch 为 None 时，计算 yaw 的平均值
                # valid_yaws = [y for y in yaw_history if y is not None]
                # if valid_yaws:  # 确保有有效数据
                #     yaw_avg = sum(valid_yaws) / len(valid_yaws)
                #     # 根据平均值正负决定电机转向
                #     yaw_angle = 5.0 if yaw_avg > 0 else -5.0  # 固定小角度旋转防止绕线
                #     try:
                #         stepper_yaw.emm_v5_move_to_angle(angle_deg=yaw_angle, vel_rpm=1, acc=10, abs_mode=False)
                #         time.sleep(0.01)
                #     except Exception:
                #         pass
                # continue
            
            if yaw != 0:
                try:
                    stepper_yaw.emm_v5_move_to_angle(angle_deg=yaw, vel_rpm=1, acc=0, abs_mode=False)
                    time.sleep(0.01)
                except Exception:
                    pass
            if pitch != 0:
                try:
                    stepper_pitch.emm_v5_move_to_angle(angle_deg=pitch, vel_rpm=1, acc=0, abs_mode=False)
                    time.sleep(0.01)
                except Exception:
                    pass
        time.sleep(0.001)  # 减少 CPU 占用

def main():
    init_board()
    if not cam.cam.isOpened():
        print("Camera open failed")
        return

    # 共享队列和标志
    position_queue = multiprocessing.Queue(maxsize=1)  # 限制队列大小为 1，避免积压
    dt_queue = multiprocessing.Queue(maxsize=1)  # 用于传递 dt
    running = multiprocessing.Event()
    running.set()

    # 启动跟踪和转向进程
    process = multiprocessing.Process(target=tracking_and_steering_thread, args=(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running))
    process.start()

    last_time = time.time()
    last_frame_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            ret, frame = cam.cam.read()
            if not ret:
                print("Failed to read frame")
                break

            update_hsv()
            if detector.task == 2:
                position = detector.task2(frame)
                if position_queue.full():
                    position_queue.get()  # 清除旧数据
                position_queue.put(position)  # 将 position 传递给进程
                result = detector.display(frame)
                cv2.imshow('Mask', detector.board_mask)
                cv2.imshow('Result', result)

            current_time = time.time()
            frame_count += 1
            dt = current_time - last_frame_time
            if dt_queue.full():
                dt_queue.get()  # 清除旧 dt
            dt_queue.put(dt)  # 将 dt 放入队列
            last_frame_time = current_time

            elapsed_time = current_time - last_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                last_time = current_time
                print(f"FPS: {fps:.2f}")

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        running.clear()  # 停止进程
        process.join()  # 等待进程结束
        cam.cam.release()
        stepper_yaw.close()
        stepper_pitch.close()
        cv2.destroyAllWindows()

cam = camera.Camera(index=0, format='MJPG', width=640, height=480, fps=240)
detector = Detector.Detector(board_color=[(13, 255, 152), (0, 51, 110)], board_min_area=18310, board_max_area=50000, diameter_ratio=0.5)
tracker = Tracker.Tracker(img_width=640, img_height = 480, vfov=100, yaw_pid = 0.03, pitch_pid = 0.03, use_kf = False, frame_add = 0, yaw_tol = 1, pitch_tol = 1)
stepper_yaw = Stepper.MotorController(port='/dev/ttyS1', baudrate=115200, timeout=0.001, motor_id=1)
stepper_pitch = Stepper.MotorController(port='/dev/ttyS3', baudrate=115200, timeout=0.001, motor_id=2)

if __name__ == "__main__":
    main()