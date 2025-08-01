import cv2
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import threading
import time
from collections import deque
import queue

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
    cv2.createTrackbar('cx_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('cy_offset', 'Controls', 30, 60, nothing)
    cv2.createTrackbar('show', 'Controls', 0, 1, nothing)
    cv2.createTrackbar('task', 'Controls', 0, 1, nothing)

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
    
    if yaw_pid != 0:
        tracker.yaw_pid = yaw_pid / 1000
    if pitch_pid != 0:
        tracker.pitch_pid = pitch_pid / 1000


def tracking_and_steering_thread(tracker, position_queue, dt_queue, running):
    """Tracking thread for testing (stepper motors disabled)
    :param tracker: Tracker instance
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
                
                # Simulate stepper motor output by printing yaw and pitch values
                print(f"Simulated yaw: {yaw:.2f}, pitch: {pitch:.2f}")
                
        except Exception as e:
            print(f"Tracking thread error: {str(e)}")
        
        time.sleep(0.001)  # Reduce CPU usage

def main():
    init_board()
    if not cam.cam.isOpened():
        print("Camera open failed")
        return

    # Shared queue and flag
    position_queue = queue.Queue(maxsize=1)  # Limit queue size to 1
    dt_queue = queue.Queue(maxsize=1)  # For dt
    running = threading.Event()
    running.set()

    # Start tracking thread (no stepper motors)
    tracking_thread = threading.Thread(target=tracking_and_steering_thread, 
                                     args=(tracker, position_queue, dt_queue, running))
    tracking_thread.daemon = True
    tracking_thread.start()

    dt = 1/30
    last_time = time.time()
    last_frame_time = time.time()
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
                    position_queue.get()  # Clear old data
                position_queue.put(position)  # Pass position data to thread

                if detector.show_img == 1:
                    result = detector.display(frame)
                    cv2.imshow('Mask', detector.board_mask)
                    cv2.imshow('Result', result)

                current_time = time.time()
                frame_count += 1
                dt = current_time - last_frame_time
                if dt_queue.full():
                    dt_queue.get()  # Clear old dt
                dt_queue.put(dt)  # Put dt in queue
                last_frame_time = current_time

                elapsed_time = current_time - last_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    last_time = current_time
                    print(f"FPS: {fps:.2f}")

                if cv2.waitKey(1) == ord('q'):
                    break

            except Exception as e:
                print(f"Main loop error: {str(e)}")

    except Exception as e:
        print(f"Main program error: {str(e)}")
        
    finally:
        running.clear()  # Stop all threads
        tracking_thread.join()  # Wait for tracking thread to end
        cam.cam.release()
        cv2.destroyAllWindows()

cam = camera.Camera(index=0, format='MJPG', width=640, height=480, fps=30)
detector = Detector.Detector(board_color=[(13, 255, 152), (0, 51, 110)], board_min_area=18310, board_max_area=50000, diameter_ratio=0.5)
tracker = Tracker.Tracker(img_width=640, img_height=480, vfov=100, yaw_pid=0.003, pitch_pid=0.003, use_kf=False, frame_add=20, shoot_tol=5, ref_point=(0.04, 0, 0))

if __name__ == "__main__":
    main()