import cv2
import numpy as np
import cam as camera

# Initialize camera
cam = camera.Camera(index=4, format='MJPG', width=1280, height=720, fps=120)

# Callback function for trackbars (empty since we get values directly)
def nothing(x):
    pass

def main():
    if not cam.cam.isOpened():
        print("Camera open failed")
        return

    # Create windows
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 1280, 720)
    cv2.resizeWindow('Mask', 1280, 720)

    # Create trackbars for HSV thresholds (initial values for light yellow)
    cv2.namedWindow('Controls')
    cv2.createTrackbar('H Min', 'Controls', 20, 179, nothing)  # Hue min (yellow ~20-30)
    cv2.createTrackbar('H Max', 'Controls', 30, 179, nothing)  # Hue max
    cv2.createTrackbar('S Min', 'Controls', 100, 255, nothing) # Saturation min
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing) # Saturation max
    cv2.createTrackbar('V Min', 'Controls', 100, 255, nothing) # Value min
    cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing) # Value max

    while True:
        ret, frame = cam.cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get trackbar positions
        h_min = cv2.getTrackbarPos('H Min', 'Controls')
        h_max = cv2.getTrackbarPos('H Max', 'Controls')
        s_min = cv2.getTrackbarPos('S Min', 'Controls')
        s_max = cv2.getTrackbarPos('S Max', 'Controls')
        v_min = cv2.getTrackbarPos('V Min', 'Controls')
        v_max = cv2.getTrackbarPos('V Max', 'Controls')

        # Create HSV threshold range
        lower_yellow = np.array([h_min, s_min, v_min])
        upper_yellow = np.array([h_max, s_max, v_max])

        # Create mask and apply it
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Display frames
        cv2.imshow('Camera', frame)
        cv2.imshow('Mask', mask)

        # Print current HSV values for debugging
        print(f"HSV Range: H({h_min}-{h_max}), S({s_min}-{s_max}), V({v_min}-{v_max})")

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cam.cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()