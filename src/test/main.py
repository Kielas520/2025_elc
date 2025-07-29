import cv2
import numpy as np
import model.cam as camera
import model.detector as Detector
import model.tracker_angle as Tracker
import model.serial as Serial
import time

def nothing(x):
    pass

def init_board():
    # Create a full-screen window
    cv2.namedWindow('Main', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Create trackbars in the main window
    cv2.createTrackbar('H Min', 'Main', 133, 179, nothing)
    cv2.createTrackbar('H Max', 'Main', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Main', 255, 255, nothing)
    cv2.createTrackbar('S Max', 'Main', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Main', 6, 255, nothing)
    cv2.createTrackbar('V Max', 'Main', 255, 255, nothing)
    cv2.createTrackbar('light_area', 'Main', 5, 200, nothing)
    cv2.createTrackbar('board_min_area', 'Main', 18310, 307200, nothing)
    cv2.createTrackbar('board_max_area', 'Main', 50000, 307200, nothing)
    cv2.createTrackbar('bin_min', 'Main', 9, 255, nothing)
    cv2.createTrackbar('bin_max', 'Main', 255, 255, nothing)
    cv2.createTrackbar('kernel_x', 'Main', 3, 10, nothing)
    cv2.createTrackbar('kernel_y', 'Main', 3, 10, nothing)
    cv2.createTrackbar('shrink', 'Main', 15, 100, nothing)
    cv2.createTrackbar('min_pic_area', 'Main', 150, 2450, nothing)
    cv2.createTrackbar('max_pic_area', 'Main', 2300, 2450, nothing)

def update_hsv():
    # Get trackbar positions
    h_min = cv2.getTrackbarPos('H Min', 'Main')
    h_max = cv2.getTrackbarPos('H Max', 'Main')
    s_min = cv2.getTrackbarPos('S Min', 'Main')
    s_max = cv2.getTrackbarPos('S Max', 'Main')
    v_min = cv2.getTrackbarPos('V Min', 'Main')
    v_max = cv2.getTrackbarPos('V Max', 'Main')
    light_area = cv2.getTrackbarPos('light_area', 'Main')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Main')
    board_max_area = cv2.getTrackbarPos('board_max_area', 'Main')
    bin_min = cv2.getTrackbarPos('bin_min', 'Main')
    bin_max = cv2.getTrackbarPos('bin_max', 'Main')
    kernel_x = cv2.getTrackbarPos('kernel_x', 'Main')
    kernel_y = cv2.getTrackbarPos('kernel_y', 'Main')
    shrink = cv2.getTrackbarPos('shrink', 'Main')
    min_pic_area = cv2.getTrackbarPos('min_pic_area', 'Main')
    max_pic_area = cv2.getTrackbarPos('max_pic_area', 'Main')

    # Update detector parameters
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

def create_control_panel(width=400, height=1080):
    """Create a black image with text labels for the controls"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add text labels for each control
    y_pos = 30
    cv2.putText(panel, "Controls:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 40
    
    controls = [
        "H Min", "H Max", "S Min", "S Max", "V Min", "V Max",
        "light_area", "board_min_area", "board_max_area",
        "bin_min", "bin_max", "kernel_x", "kernel_y",
        "shrink", "min_pic_area", "max_pic_area"
    ]
    
    for control in controls:
        cv2.putText(panel, control, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 30
    
    return panel

def ensure_3d(img):
    """Convert 2D grayscale image to 3D by duplicating channels"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def main():
    init_board()
    if not cam.cam.isOpened():
        print("Camera open failed")
        return
    
    last_time = time.time()
    frame_count = 0
    fps = 0
    
    # Define panel dimensions
    control_width = 400
    image_width = 1520  # 1920 - 400
    panel_height = 1080
    
    while True:
        ret, frame = cam.cam.read()
        if not ret:
            print("Failed to read frame")
            break

        update_hsv()
        
        position = detector.detect(frame)
        yaw, pitch = tracker.track(position, dt=1/120)
        result = detector.display(frame)
        
        # Create control panel
        control_panel = create_control_panel(width=control_width, height=panel_height)
        
        # Ensure all images are 3D (color) and resize them
        frame = ensure_3d(frame)
        mask = ensure_3d(detector.mask)
        binary = ensure_3d(detector.binary)
        result = ensure_3d(result)
        
        # Calculate individual image height
        num_images = 4  # frame, mask, binary, result
        if detector.board_img is not None:
            num_images += 1
        img_height = panel_height // num_images
        
        # Resize all images
        frame = cv2.resize(frame, (image_width, img_height))
        mask = cv2.resize(mask, (image_width, img_height))
        binary = cv2.resize(binary, (image_width, img_height))
        result = cv2.resize(result, (image_width, img_height))
        
        # Combine images vertically on the right side
        right_panel = np.vstack((frame, mask, binary, result))
        
        # If board_img exists, add it to the layout
        if detector.board_img is not None:
            board_img = ensure_3d(detector.board_img)
            board_img = cv2.resize(board_img, (image_width, img_height))
            right_panel = np.vstack((right_panel, board_img))
        
        # Resize right panel to match control panel height
        right_panel = cv2.resize(right_panel, (image_width, panel_height))
        
        # Combine controls and right panel
        composite = np.hstack((control_panel, right_panel))
        
        # Display the composite image
        cv2.imshow('Main', composite)
        
        # Calculate FPS
        current_time = time.time()
        frame_count += 1
        elapsed_time = current_time - last_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            last_time = current_time
            print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Initialize your components
cam = camera.Camera(index=4, format='MJPG', width=640, height=480, fps=240)
detector = Detector.Detector(color=[(13, 255, 152), (0, 51, 110)],
                           light_min_area=5,
                           board_min_area=18310,
                           board_max_area=50000,
                           bin_min=50, bin_max=150,
                           kernel_x=3,
                           kernel_y=3,
                           shrink_distance=15,
                           min_pic_area=150,
                           max_pic_area=2300)
tracker = Tracker.Tracker(vfov=120)

# Uncomment if using serial
# serial = Serial.Serial(port='/dev/ttyS1', baudrate=115200, timeout=1, write_timeout=1)

main()