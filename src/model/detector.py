import cv2
import numpy as np

class Light:
    def __init__(self):
        self.position = None
        self.area = None


class Detector:
    def __init__(self, color, light_min_area, kernel_x, kernel_y):
        self.bgr_upper = color[0]
        self.bgr_lower = color[1]
        self.light_min_area = light_min_area
        self.mask = None
        self.lights = []

        self.kernel_x = kernel_x
        self.kernel_y = kernel_y

        self.result_img = None
    
    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.bgr_lower, self.bgr_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_x, self.kernel_y))
        # 腐蚀
        mask = cv2.erode(mask, kernel, iterations=1)
        self.mask = mask

        return mask
 
    def find_light(self, mask):
        lights = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.light_min_area:
                x, y, _, _ = cv2.boundingRect(contour)
                light = Light()
                light.position = (x, y)
                light.area = area
                lights.append(light)
        
        self.lights = lights
        return lights



    def tf_point(self, point, frame):
        '''
        转换坐标原点，让原点变成图像中心位置
        '''
        points = []
        if frame is None:
            raise ValueError("No frame available for coordinate transformation")
        if point:
            for poin in point:
                height, width = frame.shape[:2]
                (x, y) = poin.position
                x = x - width / 2
                y = y - height / 2
                poin.position = (x, y)
                points.append(poin)
            return points
        else:
            return points

    def display(self, frame):
        img = frame.copy()  # Create a copy for drawing
        height, width = frame.shape[:2]
        # 绘制激光点（绿色）
        for light in self.lights:
            if light.position:

                abs_x = int(light.position[0] + width / 2)
                abs_y = int(light.position[1] + height / 2)
                
                cv2.circle(img, (abs_x, abs_y), 5, (0, 255, 0), -1)
        
        self.result_img = img
        return img
    
    def detect(self, frame):
        mask= self.process(frame)
        lights = self.find_light(mask)
        lights = self.tf_point(lights, frame)
        return lights