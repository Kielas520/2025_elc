import cv2
import numpy as np
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import model.steeper as Steeper
import multiprocessing
import time

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
    cv2.createTrackbar('light_area', 'Controls', 5, 200, nothing)
    cv2.createTrackbar('board_min_area', 'Controls', 18310, 307200, nothing)
    cv2.createTrackbar('board_max_area', 'Controls', 50000, 307200, nothing)
    cv2.createTrackbar('separate', 'Controls', 5, 80, nothing)
    cv2.createTrackbar('yaw_pid', 'Controls', 40, 100, nothing)
    cv2.createTrackbar('pitch_pid', 'Controls', 40, 100, nothing)

def update_hsv():
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Controls')
    board_max_area = cv2.getTrackbarPos('board_max_area', 'Controls')

    separate = cv2.getTrackbarPos('separate', 'Controls')
    yaw_pid = cv2.getTrackbarPos('yaw_pid', 'Controls')
    pitch_pid = cv2.getTrackbarPos('pitch_pid', 'Controls')

    detector.board_lower = (h_min, s_min, v_min)
    detector.board_upper = (h_max, s_max, v_max)

    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    
    if yaw_pid == 0:
        pass
    else:
        tracker.yaw_pid = yaw_pid / 100
    if pitch_pid == 0:
        pass
    else:
        tracker.pitch_pid = pitch_pid / 100

    if separate == 0:
        detector.separate = 1
    else:
        detector.separate = separate

def tracking_and_steering_thread(tracker, steeper, position_queue, running, dt):
    """跟踪和转向线程
    :param tracker: Tracker 实例
    :param steeper: Steeper.MotorController 实例
    :param position_queue: 共享队列，接收 detector 的 position 数据
    :param running: 运行标志
    :param dt: 时间增量
    """
    while running.is_set():
        if not position_queue.empty():
            position = position_queue.get()
            yaw, pitch = tracker.track1(position, dt)
            if yaw != 0:  # 仅当 yaw 不为 0 时移动
                try:
                    steeper.emm_v5_move_to_angle(angle_deg=yaw, vel_rpm=100, acc=100, abs_mode=False)
                    time.sleep(0.01)  # 短暂延迟，避免过于频繁发送
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
    running = multiprocessing.Event()
    running.set()

    # 启动跟踪和转向线程
    thread = multiprocessing.Process(target=tracking_and_steering_thread, args=(tracker, steeper_yaw, position_queue, running, 1/30))
    thread.start()

    last_time = time.time()
    frame_count = 0
    fps = 0
    dt = 1 / 30

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
                position_queue.put(position)  # 将 position 传递给线程
                result = detector.display(frame)
                cv2.imshow('Mask', detector.board_mask)
                cv2.imshow('Result', result)

            current_time = time.time()
            frame_count += 1
            elapsed_time = current_time - last_time
            dt = elapsed_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                last_time = current_time
                print(f"FPS: {fps:.2f}")

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        running.clear()  # 停止线程
        thread.join()  # 等待线程结束
        cam.cam.release()
        steeper_yaw.close()
        cv2.destroyAllWindows()

cam = camera.Camera(index=0, format='MJPG', width=1280, height=720, fps=30)
detector = Detector.Detector(board_color=[(13, 255, 152), (0, 51, 110)], board_min_area=18310, board_max_area=50000)
tracker = Tracker.Tracker(img_width=1280, vfov=100)
steeper_yaw = Steeper.MotorController(port='/dev/ttyUSB0', baudrate=115200, timeout=0.001, motor_id=1)

if __name__ == "__main__":
    main()