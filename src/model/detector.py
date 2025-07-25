import cv2
import numpy as np

class Light:
    def __init__(self):
        self.center = None
        self.area = None

class Board:
    def __init__(self):
        self.points = []

class Detector:
    def __init__(self, color, light_min_area, board_min_area, bin_val):
        self.bgr_upper = color[0]
        self.bgr_lower = color[1]
        self.light_min_area = light_min_area
        self.mask = None
        self.lights = []

        self.binary = None
        self.bin_val = bin_val
        self.board_min_area = board_min_area
        self.boards = []

        self.result_img = None
    
    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.bgr_lower, self.bgr_upper)
        self.mask = mask

        # 背景板检测（灰度 + Otsu）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.bin_val, 255, cv2.THRESH_BINARY)
        self.binary = binary
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask = cv2.dilate(mask, kernel, iterations=1)

        return mask, binary
    
    def find_board(self, binary):
        boards = []
        board_contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in board_contours:
            area = cv2.contourArea(contour)
            if area > self.board_min_area:
                # 逼近多边形，获取四边形
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 筛选四边形
                if len(approx) == 4:
                    # 获取四个角点
                    points = approx.reshape(4, 2)
                    
                    # 按左上、左下、右下、右上排序
                    sum_xy = points.sum(axis=1)
                    diff_xy = points[:, 0] - points[:, 1]
                    sorted_points = [
                        points[np.argmin(sum_xy)],  # 左上：x+y 最小
                        points[np.argmax(diff_xy)],  # 左下：x-y 最大
                        points[np.argmax(sum_xy)],  # 右下：x+y 最大
                        points[np.argmin(diff_xy)]   # 右上：x-y 最小
                    ]
                    
                    # 创建 Board 对象
                    board = Board()
                    board.points = [tuple(pt) for pt in sorted_points]
                    boards.append(board)
        
        self.boards = boards
        return boards
    
    def find_light(self, mask, img):
        lights = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.light_min_area:
                x, y, w, h = cv2.boundingRect(contour)
                light = Light()
                center_x = x + w / 2 - width / 2
                center_y = y + h / 2 - height / 2
                light.center = (center_x, center_y)
                light.area = area
                lights.append(light)
        
        self.lights = lights
        return lights
    
    def display(self, frame):
        img = frame.copy()  # Create a copy for drawing
        # 绘制背景板（四边形连线）
        for board in self.boards:
            if len(board.points) == 4:
                pts = np.array(board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                # 可选：标记角点序号
                for i, pt in enumerate(board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制激光点（中心点）
        for light in self.lights:
            if light.center:
                # 相对坐标转绝对坐标
                center_x = int(light.center[0] + img.shape[1] / 2)
                center_y = int(light.center[1] + img.shape[0] / 2)
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
        self.result_img = img
        return img
    
    def detect(self, frame):
        mask, binary = self.process(frame)
        lights = self.find_light(mask, frame)
        boards = self.find_board(binary)
        return lights