import cv2
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import model.stepper as Stepper
from model.status import GPIN
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
    cv2.createTrackbar('yaw_pid', 'Controls', 40, 3000, nothing)
    cv2.createTrackbar('pitch_pid', 'Controls', 40, 3000, nothing)
    cv2.createTrackbar('yaw_tol', 'Controls', 1, 10, nothing)
    cv2.createTrackbar('pitch_tol', 'Controls', 1, 10, nothing)
    cv2.createTrackbar('task', 'Controls', 0, 1, nothing)
    cv2.createTrackbar('cx_offset', 'Controls', -30, 30, nothing)
    cv2.createTrackbar('cy_offset', 'Controls', -30, 30, nothing)

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
    task = cv2.getTrackbarPos('task', 'Controls')
    cx_offset = cv2.getTrackbarPos('cx_offset', 'Controls')
    cy_offset = cv2.getTrackbarPos('cy_offset', 'Controls')

    detector.board_lower = (h_min, s_min, v_min)
    detector.board_upper = (h_max, s_max, v_max)
    detector.diameter_ratio = diameter_ratio / 100
    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    detector.task = task
    detector.cx_offset = cx_offset
    detector.cy_offset = cy_offset

    if yaw_pid != 0:
        tracker.yaw_pid = yaw_pid / 1000
    if pitch_pid != 0:
        tracker.pitch_pid = pitch_pid / 1000

    tracker.yaw_tol = yaw_tol
    tracker.pitch_tol = pitch_tol

def tracking_and_steering_thread(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running):
    """Tracking and steering thread
    :param tracker: Tracker instance
    :param stepper_yaw: Stepper.MotorController instance (yaw)
    :param stepper_pitch: Stepper.MotorController instance (pitch)
    :param position_queue: Shared queue for detector position data
    :param dt_queue: Shared queue for dt data
    :param running: Running flag
    """
    yaw_history = deque(maxlen=4)  # Store last 4 yaw values
    pitch_history = deque(maxlen=4)  # Store last 4 pitch values

    while running.is_set():
        try:
            dt = 1/30  # Default value
            if not dt_queue.empty():
                dt = dt_queue.get()  # Get latest dt

            if not position_queue.empty():
                position = position_queue.get()
                yaw, pitch = tracker.track(position, dt)
                
                yaw_history.append(yaw)
                pitch_history.append(pitch)

                if yaw is None or pitch is None:
                    yaw = 0
                    pitch = 0
                    continue
                
                if yaw != 0:
                    try:
                        stepper_yaw.emm_v5_move_to_angle(angle_deg=yaw, vel_rpm=1, acc=0, abs_mode=False)
                        time.sleep(0.01)
                    except Exception as e:
                        print(f"Yaw motor error: {str(e)}")
                
                if pitch != 0:
                    try:
                        stepper_pitch.emm_v5_move_to_angle(angle_deg=pitch, vel_rpm=1, acc=0, abs_mode=False)
                        time.sleep(0.01)
                    except Exception as e:
                        print(f"Pitch motor error: {str(e)}")
                        
        except Exception as e:
            print(f"Tracking thread error: {str(e)}")
        
        time.sleep(0.001)  # Reduce CPU usage

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
            # 心跳 LED 闪烁
            heart_beat.flash()  # 使用 flash 方法，约 1 秒周期
            time.sleep(0.05)    # 20 * 0.05s = 1s 闪烁周期

            # 显示 detector 的 task 属性
            if detector.task in [0, 1]:  # 确保任务值有效
                task_info.set_value(1 if detector.task > 0 else 0)  # 0 输出低电平，非 0 输出高电平

            # 检查任务切换输入
            new_task = task_switch.button_callback(detector.task)
            if new_task is not None:  # 检测到任务切换
                detector.task = new_task  # 更新 detector.task
                time.sleep(0.1)  # 防抖延时
            
            if tracker.shoot == 0:
                lazer.set_value(0)

            elif tracker.shoot == 1:
                lazer.set_value(1)


    except Exception as e:
        print(f"决策线程错误: {str(e)}")
    finally:
        # 清理 GPIO 引脚状态
        heart_beat.set_value(0)
        task_info.set_value(0)

def main():
    init_board()
    if not cam.cam.isOpened():
        print("Camera open failed")
        return

    # 共享队列和标志
    position_queue = queue.Queue(maxsize=1)  # 限制队列大小为 1
    dt_queue = queue.Queue(maxsize=1)  # 用于 dt
    running = threading.Event()
    running.set()

    # 启动跟踪和转向线程
    tracking_thread = threading.Thread(target=tracking_and_steering_thread, 
                                     args=(tracker, stepper_yaw, stepper_pitch, position_queue, dt_queue, running))
    tracking_thread.daemon = True
    tracking_thread.start()

    # 启动决策线程
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
                    position = detector.task2(frame)
                
                if position_queue.full():
                    position_queue.get()  # 清除旧数据
                position_queue.put(position)  # 将位置数据传递给线程

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

                beat += 1
                beat %= 255

                if cv2.waitKey(1) == ord('q'):
                    break

            except Exception as e:
                print(f"主循环错误: {str(e)}")

    except Exception as e:
        print(f"主程序错误: {str(e)}")
        
    finally:
        running.clear()  # 停止所有线程
        tracking_thread.join()  # 等待跟踪线程结束
        decision_thread.join()  # 等待决策线程结束
        cam.cam.release()
        try:
            stepper_yaw.close()
        except Exception as e:
            print(f"关闭 yaw 步进电机错误: {str(e)}")
        try:
            stepper_pitch.close()
        except Exception as e:
            print(f"关闭 pitch 步进电机错误: {str(e)}")
        GPIO.cleanup()  # 清理所有 GPIO 引脚
        cv2.destroyAllWindows()


cam = camera.Camera(index=0, format='MJPG', width=720, height=480, fps=240)

detector = Detector.Detector(board_color=[(13, 255, 152), (0, 51, 110)], board_min_area=18310, board_max_area=50000, diameter_ratio=0.5)
tracker = Tracker.Tracker(img_width=640, img_height=480, vfov=100, yaw_pid=0.03, pitch_pid=0.03, use_kf=False, frame_add=0, yaw_tol=1, pitch_tol=1)

stepper_yaw = Stepper.MotorController(port='/dev/ttyS1', baudrate=115200, timeout=0.001, motor_id=1)
stepper_pitch = Stepper.MotorController(port='/dev/ttyS3', baudrate=115200, timeout=0.001, motor_id=2)

heart_beat = GPIN(pin = 11, mode = 0)
task_info = GPIN(pin = 13, mode = 0)
task_switch = GPIN(pin = 15, mode = 1)
lazer = GPIN(pin = 16, mode = 0)

if __name__ == "__main__":
    import queue  # Moved import here to avoid global scope issues
    main()