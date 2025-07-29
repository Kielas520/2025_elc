import cv2
import numpy as np
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import model.serial as Serial
import time  # 导入 time 模块

def nothing(x):
    pass

def init_board():
    cv2.namedWindow('Result', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('Mask', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('board', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('binary', cv2.WINDOW_FREERATIO)
    cv2.moveWindow('Mask', 360, 180)
    cv2.moveWindow('binary', 360, 0)
    cv2.moveWindow('Result', 540, 180)
    cv2.moveWindow('board', 540, 0)

    cv2.resizeWindow('Mask', 170, 150)
    cv2.resizeWindow('binary', 170, 150)
    cv2.resizeWindow('Result', 170,150)
    cv2.resizeWindow('board', 170, 150)

    # Create trackbars for HSV thresholds (initial values for light yellow)
    cv2.namedWindow('Controls', cv2.WINDOW_FREERATIO)
    cv2.moveWindow('Controls', 0, 0)
    cv2.resizeWindow('Controls', 320, 400)  # 增大窗口尺寸
    cv2.createTrackbar('H Min', 'Controls', 133, 179, nothing)  # Hue min (yellow ~20-30)
    cv2.createTrackbar('H Max', 'Controls', 179, 179, nothing)  # Hue max
    cv2.createTrackbar('S Min', 'Controls', 255, 255, nothing) # Saturation min
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing) # Saturation max
    cv2.createTrackbar('V Min', 'Controls', 6, 255, nothing) # Value min
    cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing) # Value max
    cv2.createTrackbar('light_area', 'Controls', 5, 200, nothing)
    cv2.createTrackbar('board_min_area', 'Controls', 18310, 307200, nothing)
    cv2.createTrackbar('board_max_area', 'Controls', 50000, 307200, nothing)
    cv2.createTrackbar('bin_min', 'Controls', 9, 255, nothing)
    cv2.createTrackbar('bin_max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('kernel_x', 'Controls', 3, 10, nothing)
    cv2.createTrackbar('kernel_y', 'Controls', 3, 10, nothing)
    cv2.createTrackbar('shrink', 'Controls', 15, 100, nothing)
    cv2.createTrackbar('min_pic_area', 'Controls', 150, 2450, nothing)
    cv2.createTrackbar('max_pic_area', 'Controls', 2300, 2450, nothing)

def update_hsv():
    # Get trackbar positions
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    light_area = cv2.getTrackbarPos('light_area', 'Controls')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Controls')
    board_max_area = cv2.getTrackbarPos('board_max_area', 'Controls')
    bin_min = cv2.getTrackbarPos('bin_min', 'Controls')
    bin_max = cv2.getTrackbarPos('bin_max', 'Controls')
    kernel_x = cv2.getTrackbarPos('kernel_x', 'Controls')
    kernel_y = cv2.getTrackbarPos('kernel_y', 'Controls')
    shrink = cv2.getTrackbarPos('shrink', 'Controls')
    min_pic_area = cv2.getTrackbarPos('min_pic_area', 'Controls')
    max_pic_area = cv2.getTrackbarPos('max_pic_area', 'Controls')

    # Create HSV threshold range
    detector.bgr_lower = (h_min, s_min, v_min)
    detector.bgr_upper = (h_max, s_max, v_max)
    detector.light_min_area = light_area
    detector.board_min_area = board_min_area
    detector.board_max_area = board_max_area
    detector.bin_min = bin_min
    detector.bin_max = bin_max
    detector.kernel_x = kernel_x
    detector.kernel_y = kernel_y
    detector.shrink_distance = shrink
    detector.min_pic_area = min_pic_area
    detector.max_pic_area = max_pic_area


def main():
    init_board()
    if not cam.cam.isOpened():
        print("Camera open failed")
        return
    last_time = time.time()  # 记录上一帧时间
    frame_count = 0  # 帧计数
    fps = 0  # 初始化 FPS
    while True:
        ret, frame = cam.cam.read()
        if not ret:
            print("Failed to read frame")
            break

        update_hsv()
        
        position = detector.detect(frame)
        yaw, pitch = tracker.track(position, dt=1/120)  # 传递 dt 参数给 tracker
        #print(yaw,pitch)
        result = detector.display(frame)
        serial.send_data(yaw = -yaw * 0.05, pitch = pitch * 0.05)
        if detector.board_img is not None:
            cv2.imshow('board',detector.board_img)
        cv2.imshow('Mask', detector.mask)
        cv2.imshow('binary', detector.binary)
        cv2.imshow('Result', result)

        # 计算 FPS
        current_time = time.time()
        frame_count += 1
        elapsed_time = current_time - last_time
        if elapsed_time >= 1.0:  # 每秒更新一次 FPS
            fps = frame_count / elapsed_time
            frame_count = 0
            last_time = current_time
            print(f"FPS: {fps:.2f}")  # 打印 FPS，保留两位小数

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

cam = camera.Camera(index=0
                    , format='MJPG'
                    , width=640
                    , height=480
                    , fps=240)

detector = Detector.Detector(color = [(13, 255, 152), (0, 51, 110)]
                             , light_min_area = 5
                             , board_min_area = 18310
                             , board_max_area = 50000
                             , bin_min = 50, bin_max = 150
                             , kernel_x = 3
                             , kernel_y = 3
                             , shrink_distance = 15
                             , min_pic_area = 150
                             , max_pic_area = 2300)

tracker = Tracker.Tracker(vfov = 120)

serial = Serial.Serial(port='/dev/ttyS1'
, baudrate=115200
, timeout=1
, write_timeout=1)
main()