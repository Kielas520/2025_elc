import cv2
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import model.stepper as Stepper
from model.status import GPIN
import queue
import threading
import time
from collections import deque
import Hobot.GPIO as GPIO

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
    cv2.resizeWindow('Controls', 320, 500)
    cv2.createTrackbar('H Min', 'Controls', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'Controls', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Controls', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Controls', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Controls', 55, 255, nothing)
    cv2.createTrackbar('board_min_area', 'Controls', 5000, 307200, nothing)
    cv2.createTrackbar('board_max_area', 'Controls', 54000, 307200, nothing)
    cv2.createTrackbar('diameter_ratio', 'Controls', 40, 70, nothing)
    cv2.createTrackbar('yaw_kp', 'Controls', 5, 100, nothing)  # 偏航角倍率: 0-10
    cv2.createTrackbar('pitch_kp', 'Controls', 6, 100, nothing)  # 俯仰角倍率: 0-10
    cv2.createTrackbar('vel_rpm', 'Controls', 5000, 5000, nothing)  # 速度: 0-5000 RPM
    cv2.createTrackbar('acc', 'Controls', 255, 255, nothing)  # 加速度: 0-255
    cv2.createTrackbar('cx_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('cy_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('show', 'Controls', 0, 1, nothing)
    cv2.createTrackbar('shoot_tol', 'Controls', 14, 200, nothing)
    cv2.createTrackbar('offset_pitch', 'Controls', 11, 20, nothing)

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
    yaw_kp = cv2.getTrackbarPos('yaw_kp', 'Controls')
    pitch_kp = cv2.getTrackbarPos('pitch_kp', 'Controls')
    vel_rpm = cv2.getTrackbarPos('vel_rpm', 'Controls')
    acc = cv2.getTrackbarPos('acc', 'Controls')
    cx_offset = cv2.getTrackbarPos('cx_offset', 'Controls')
    cy_offset = cv2.getTrackbarPos('cy_offset', 'Controls')
    show = cv2.getTrackbarPos('show', 'Controls')
    shoot_tol = cv2.getTrackbarPos('shoot_tol', 'Controls')
    offset_pitch = cv2.getTrackbarPos('offset_pitch', 'Controls')

    detector.board_lower = (h_min, s_min, v_min)
    detector.board_upper = (h_max, s_max, v_max)
    detector.diameter_ratio = diameter_ratio / 100
    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    detector.cx_offset = cx_offset - 30
    detector.cy_offset = cy_offset - 30
    detector.show_img = show
    tracker.offset_pitch = offset_pitch - 10
    tracker.shoot_tol = shoot_tol

    return yaw_kp / 100.0, pitch_kp / 100.0, vel_rpm, acc

def tracking_and_steering_thread(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running):
    """跟踪和转向线程，使用 kp 倍率控制"""
    while running.is_set():
        try:
            dt = 1/30
            if not dt_queue.empty():
                dt = dt_queue.get()

            if not position_queue.empty():
                position = position_queue.get()
                target_yaw, target_pitch = tracker.track(position, dt)
                if tracker.if_lost == True:
                    try:
                        stepper_yaw.emm_v5_move_to_angle(angle_deg=3, vel_rpm=2000, acc=0, abs_mode=False)
                    except Exception as e:
                        print(f"Yaw 电机错误: {str(e)}")
                else:
                    # 获取控制参数
                    yaw_kp, pitch_kp, vel_rpm, acc = update_hsv()
                    if abs(target_yaw) > 0:  # 避免微小调整
                        try:
                            stepper_yaw.emm_v5_move_to_angle(angle_deg=target_yaw * yaw_kp, vel_rpm=vel_rpm, acc=acc, abs_mode=False)
                        except Exception as e:
                            print(f"Yaw 电机错误: {str(e)}")
                    if abs(target_pitch) > 0:
                        try:
                            stepper_pitch.emm_v5_move_to_angle(angle_deg=target_pitch * pitch_kp, vel_rpm=vel_rpm, acc=acc, abs_mode=False)
                        except Exception as e:
                            print(f"Pitch 电机错误: {str(e)}")

        except Exception as e:
            print(f"跟踪线程错误: {str(e)}")

        # 移除 time.sleep 以提高频率
        time.sleep(0.000001)  # 可选：如果 CPU 使用率过高，可尝试 0.1ms

def decision(running, detector, tracker, heart_beat, task_info, task_switch, lazer):
    """决策线程，用于处理心跳、任务信息显示和任务切换"""
    try:
        while running.is_set():
            heart_beat.flash()
            if detector.task in [0, 1]:
                task_info.set_value(1 if detector.task > 0 else 0)
                lazer.set_value(0 if detector.task > 0 else 0)
            new_task = task_switch.button_callback(detector.task)
            detector.task = new_task
            if not tracker.shoot:
                lazer.set_value(0)
            else:
                lazer.set_value(1)
            
    except Exception as e:
        print(f"决策线程错误: {str(e)}")
    finally:
        heart_beat.set_value(0)
        task_info.set_value(0)
        lazer.set_value(0)

def main():
    init_board()
    if not cam.cam.isOpened():
        print("Camera open failed")
        return

    position_queue = queue.Queue(maxsize=1)
    dt_queue = queue.Queue(maxsize=1)
    running = threading.Event()
    running.set()

    tracking_thread = threading.Thread(target=tracking_and_steering_thread, 
                                     args=(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running))
    tracking_thread.daemon = True
    tracking_thread.start()

    decision_thread = threading.Thread(target=decision,
                                     args=(running, detector, tracker, heart_beat, task_info, task_switch, lazer))
    decision_thread.daemon = True
    decision_thread.start()

    dt = 1/30
    last_time = time.time()
    last_frame_time = time.time()
    beat = 0
    frame_count = 0
    fps = 0

    try:
        while True:
            try:
                ret, frame = cam.cam.read()
                if not ret:
                    print("Failed to read frame")
                    break

                update_hsv()

                if detector.task == 0:
                    position = detector.task1(frame)
                elif detector.task == 1:
                    position = detector.task2(frame, tracker)
                if position_queue.full():
                    position_queue.get()
                position_queue.put(position)
                if detector.show_img == 1:
                    result = detector.display(frame)
                    cv2.imshow('Mask', detector.board_mask)
                    cv2.imshow('Result', result)

                print(tracker.shoot)
                current_time = time.time()
                frame_count += 1
                dt = current_time - last_frame_time
                if dt_queue.full():
                    dt_queue.get()
                dt_queue.put(dt)
                last_frame_time = current_time

                elapsed_time = current_time - last_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    last_time = current_time
                    print(f"FPS: {fps:.2f}")
                beat += 1
                beat %= 255

                if cv2.waitKey(1) == ord('q'):
                    break

            except Exception as e:
                print(f"主循环错误: {str(e)}")

    except Exception as e:
        print(f"主程序错误: {str(e)}")
        
    finally:
        running.clear()
        tracking_thread.join()
        decision_thread.join()
        cam.cam.release()
        try:
            stepper_yaw.close()
        except Exception as e:
            print(f"关闭 yaw 步进电机错误: {str(e)}")
        try:
            stepper_pitch.close()
        except Exception as e:
            print(f"关闭 pitch 步进电机错误: {str(e)}")
        GPIO.cleanup()
        cv2.destroyAllWindows()

cam = camera.Camera(index=0, format='MJPG', width=640, height=480, fps=240)
detector = Detector.Detector(board_color=[(13, 255, 152), (0, 51, 110)], board_min_area=18310, board_max_area=50000, diameter_ratio=0.5)
tracker = Tracker.Tracker(img_width=640, img_height=480, vfov=100, use_kf=False, frame_add=50, shoot_tol=5, ref_point=(-0.03, 0, 0), offset_pitch=-10, offset_yaw=0.0, is_mirrored=False)
stepper_yaw = Stepper.MotorController(port='/dev/ttyS1', baudrate=115200, timeout=0.001, motor_id=1)
stepper_pitch = Stepper.MotorController(port='/dev/ttyS3', baudrate=115200, timeout=0.001, motor_id=2)
heart_beat = GPIN(pin=13, mode=1)
task_info = GPIN(pin=11, mode=1)
lazer = GPIN(pin=16, mode=1)
task_switch = GPIN(pin=15, mode=0)

if __name__ == "__main__":
    main()