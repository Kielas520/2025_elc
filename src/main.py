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
from pid_controller import PIDController

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
    cv2.resizeWindow('Controls', 320, 500)  # 增加窗口高度以容纳更多轨迹条
    cv2.createTrackbar('H Min', 'Controls', 133, 179, nothing)
    cv2.createTrackbar('H Max', 'Controls', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Controls', 6, 255, nothing)
    cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('board_min_area', 'Controls', 18310, 307200, nothing)
    cv2.createTrackbar('board_max_area', 'Controls', 50000, 307200, nothing)
    cv2.createTrackbar('diameter_ratio', 'Controls', 40, 70, nothing)
    cv2.createTrackbar('yaw_Kp', 'Controls', 40, 3000, nothing)  # 比例增益
    cv2.createTrackbar('yaw_Ki', 'Controls', 0, 100, nothing)    # 积分增益
    cv2.createTrackbar('yaw_Kd', 'Controls', 0, 100, nothing)    # 微分增益
    cv2.createTrackbar('pitch_Kp', 'Controls', 40, 3000, nothing)  # 比例增益
    cv2.createTrackbar('pitch_Ki', 'Controls', 0, 100, nothing)   # 积分增益
    cv2.createTrackbar('pitch_Kd', 'Controls', 0, 100, nothing)   # 微分增益
    cv2.createTrackbar('cx_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('cy_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('show', 'Controls', 0, 1, nothing)

def update_hsv(yaw_pid_controller, pitch_pid_controller):
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Controls')
    board_max_area = cv2.getTrackbarPos('board_max_area', 'Controls')
    diameter_ratio = cv2.getTrackbarPos('diameter_ratio', 'Controls')
    yaw_Kp = cv2.getTrackbarPos('yaw_Kp', 'Controls')
    yaw_Ki = cv2.getTrackbarPos('yaw_Ki', 'Controls')
    yaw_Kd = cv2.getTrackbarPos('yaw_Kd', 'Controls')
    pitch_Kp = cv2.getTrackbarPos('pitch_Kp', 'Controls')
    pitch_Ki = cv2.getTrackbarPos('pitch_Ki', 'Controls')
    pitch_Kd = cv2.getTrackbarPos('pitch_Kd', 'Controls')
    cx_offset = cv2.getTrackbarPos('cx_offset', 'Controls')
    cy_offset = cv2.getTrackbarPos('cy_offset', 'Controls')
    show = cv2.getTrackbarPos('show', 'Controls')

    detector.board_lower = (h_min, s_min, v_min)
    detector.board_upper = (h_max, s_max, v_max)
    detector.diameter_ratio = diameter_ratio / 100
    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    detector.cx_offset = cx_offset - 30
    detector.cy_offset = cy_offset - 30
    detector.show_img = show
    
    # 更新 PID 控制器的参数
    if yaw_Kp != 0:
        yaw_pid_controller.set_Kp(yaw_Kp / 1000)  # 缩放到 0-3
    yaw_pid_controller.set_Ki(yaw_Ki / 1000)      # 缩放到 0-0.1
    yaw_pid_controller.set_Kd(yaw_Kd / 1000)      # 缩放到 0-0.1
    if pitch_Kp != 0:
        pitch_pid_controller.set_Kp(pitch_Kp / 1000)  # 缩放到 0-3
    pitch_pid_controller.set_Ki(pitch_Ki / 1000)      # 缩放到 0-0.1
    pitch_pid_controller.set_Kd(pitch_Kd / 1000)      # 缩放到 0-0.1

def tracking_and_steering_thread(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running, yaw_pid_controller, pitch_pid_controller):
    """跟踪和转向线程，实现 PID 闭环控制
    :param tracker: Tracker 实例
    :param stepper_yaw: Stepper.MotorController 实例 (yaw)
    :param stepper_pitch: Stepper.MotorController 实例 (pitch)
    :param position_queue: 检测器位置数据的共享队列
    :param dt_queue: 时间步长数据的共享队列
    :param running: 运行标志
    :param yaw_pid_controller: Yaw 的 PID 控制器
    :param pitch_pid_controller: Pitch 的 PID 控制器
    """
    current_yaw_angle = 0  # 假设初始角度为 0
    current_pitch_angle = 0

    while running.is_set():
        try:
            dt = 1/30  # 默认时间步长
            if not dt_queue.empty():
                dt = dt_queue.get()  # 获取最新 dt

            if not position_queue.empty():
                position = position_queue.get()
                target_yaw, target_pitch = tracker.track(position, dt)

                if target_yaw is None or target_pitch is None:
                    target_yaw = 0
                    target_pitch = 0
                    continue

                # 计算 yaw 误差并更新 PID 控制
                yaw_error = target_yaw - current_yaw_angle
                yaw_output = yaw_pid_controller.update(yaw_error, dt)
                if abs(yaw_output) > 0.01:  # 避免微小调整
                    try:
                        current_yaw_angle += yaw_output
                        stepper_yaw.emm_v5_move_to_angle(angle_deg=yaw_output, vel_rpm=1, acc=0, abs_mode=False)
                        time.sleep(0.01)
                    except Exception as e:
                        print(f"Yaw 电机错误: {str(e)}")

                # 计算 pitch 误差并更新 PID 控制
                pitch_error = target_pitch - current_pitch_angle
                pitch_output = pitch_pid_controller.update(pitch_error, dt)
                if abs(pitch_output) > 0.01:  # 避免微小调整
                    try:
                        current_pitch_angle += pitch_output
                        stepper_pitch.emm_v5_move_to_angle(angle_deg=pitch_output, vel_rpm=1, acc=0, abs_mode=False)
                        time.sleep(0.01)
                    except Exception as e:
                        print(f"Pitch 电机错误: {str(e)}")

        except Exception as e:
            print(f"跟踪线程错误: {str(e)}")

        time.sleep(0.001)  # 减少 CPU 使用率

def decision(running, detector, tracker, heart_beat, task_info, task_switch, lazer):
    """决策线程，用于处理心跳、任务信息显示和任务切换
    :param running: 线程事件，用于控制线程执行
    :param detector: Detector 实例，用于访问和修改任务属性
    :param heart_beat: GPIN 实例，用于心跳 LED
    :param task_info: GPIN 实例，用于任务状态显示
    :param task_switch: GPIN 实例，用于任务切换输入
    """
    try:
        while running.is_set():
            heart_beat.flash()
            if detector.task in [0, 1]:
                task_info.set_value(1 if detector.task > 0 else 0)
            new_task = task_switch.button_callback(detector.task)
            detector.task = new_task
            if tracker.shoot == 0:
                lazer.set_value(0)
            elif tracker.shoot == 1:
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

    # 初始化 PID 控制器
    yaw_pid_controller = PIDController(Kp=tracker.yaw_pid, Ki=0.0, Kd=0.0, dt=1/30)
    pitch_pid_controller = PIDController(Kp=tracker.pitch_pid, Ki=0.0, Kd=0.0, dt=1/30)

    position_queue = queue.Queue(maxsize=1)
    dt_queue = queue.Queue(maxsize=1)
    running = threading.Event()
    running.set()

    tracking_thread = threading.Thread(target=tracking_and_steering_thread, 
                                     args=(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running, yaw_pid_controller, pitch_pid_controller))
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

                update_hsv(yaw_pid_controller, pitch_pid_controller)

                if detector.task == 0:
                    position = detector.task1(frame)
                elif detector.task == 1:
                    position = detector.task2(frame)
                
                if position_queue.full():
                    position_queue.get()
                position_queue.put(position)
                if detector.show_img == 1:
                    result = detector.display(frame)
                    cv2.imshow('Mask', detector.board_mask)
                    cv2.imshow('Result', result)

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
tracker = Tracker.Tracker(img_width=640, img_height=480, vfov=100, yaw_pid=0.003, pitch_pid=0.003, use_kf=False, frame_add=20, shoot_tol=5, ref_point=(-0.04, 0, 0))
stepper_yaw = Stepper.MotorController(port='/dev/ttyS1', baudrate=115200, timeout=0.001, motor_id=1)
stepper_pitch = Stepper.MotorController(port='/dev/ttyS3', baudrate=115200, timeout=0.001, motor_id=2)
heart_beat = GPIN(pin=13, mode=1)
task_info = GPIN(pin=11, mode=1)
task_switch = GPIN(pin=15, mode=0)
lazer = GPIN(pin=16, mode=1)

if __name__ == "__main__":
    main()