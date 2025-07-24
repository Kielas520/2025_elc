import cv2

class Blob:
    def __init__(self):
        self.center = None
        self.area = None

class Detector:
    def __init__(self, color, min_area):
        self.bgr_upper = color[0]
        self.bgr_lower = color[1]
        self.mask = None
        self.min_area = min_area
        self.result = None
        self.blobs = []
    
    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.bgr_lower, self.bgr_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        self.mask = mask
        return mask, result
    
    def find_color(self, mask, img):
        blobs = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 获取图像尺寸
        height, width = img.shape[:2]
            
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.circle(img, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)
                blob = Blob()
                # 计算相对于图像中心的坐标
                center_x = x + w / 2 - width / 2
                center_y = y + h / 2 - height / 2
                blob.center = (center_x, center_y)
                blob.area = area
                blobs.append(blob)

        self.blobs = blobs
        self.result = img
        return blobs
    
    def detect(self, frame):
        mask, result = self.process(frame)
        blobs = self.find_color(mask, result)
        return blobs

