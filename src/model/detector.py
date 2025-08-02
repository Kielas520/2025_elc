import cv2
import numpy as np
import math
from model.tracker import Tracker
class Board:
    def __init__(self):
        self.points = []  # 四边形角点 [左上, 左下, 右下, 右上]
        self.area = None
        self.center = None
        
class Detector:
    def __init__(self, board_color, board_min_area, board_max_area, diameter_ratio, cx_offset = 0, cy_offset = 0):
        self.std_square = np.float32([[0, 0], [30, 0], [30, 21], [0, 21]])
        self.board_lower = board_color[0]
        self.board_upper = board_color[1]

        self.task = 0

        self.board_mask = None
        self.board_min_area = board_min_area
        self.board_max_area = board_max_area
        self.board = None
        
        self.cx_offset = cx_offset
        self.cy_offset = cy_offset
        self.diameter_ratio = diameter_ratio
        self.circle_step = 0
        self.show_img = 0
        self.result_img = None
        self.target = None
        # 检查当前点与屏幕中心点的距离
        self.frame_center = None
        self.lazer_center = None

    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.board_lower, self.board_upper)
        self.board_mask = mask
        self.frame_center = (frame.shape[1] / 2 + self.cx_offset, frame.shape[0] / 2 + self.cy_offset)
        return mask

    def find_board(self, binary):
        boards = []
        # 使用 cv2.RETR_CCOMP 以获取内外轮廓的层次结构
        board_contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 首先尝试寻找内轮廓
        inner_contours = []
        for i, contour in enumerate(board_contours):
            if hierarchy[0][i][3] != -1:  # 有父轮廓的轮廓（内轮廓）
                inner_contours.append((i, contour))
        
        # 如果没有内轮廓，则使用外轮廓（无父轮廓的轮廓）
        target_contours = inner_contours if inner_contours else [(i, c) for i, c in enumerate(board_contours) if hierarchy[0][i][3] == -1]
        
        for i, contour in target_contours:
            area = cv2.contourArea(contour)
            if area > self.board_min_area and area < self.board_max_area:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    points = approx.reshape(4, 2)
                    
                    # 原有排序逻辑
                    sum_xy = points.sum(axis=1)
                    diff_xy = points[:, 0] - points[:, 1]
                    sorted_points = [
                        points[np.argmin(sum_xy)],  # 左上
                        points[np.argmax(diff_xy)], # 左下
                        points[np.argmax(sum_xy)],  # 右下
                        points[np.argmin(diff_xy)]  # 右上
                    ]
                    
                    # 检查排序后的点是否有重合
                    unique_points = set(tuple(pt) for pt in sorted_points)
                    
                    if len(unique_points) < 4:
                        # 如果有任何点重合，重新按新规则排序
                        # 新排序规则：左上最左，左下最上，右下最右，右上最下
                        sorted_points = [
                            points[np.argmin(points[:, 0])],  # 左上：最左边的x坐标
                            points[np.argmin(points[:, 1])],  # 左下：最上面的y坐标
                            points[np.argmax(points[:, 0])],  # 右下：最右边的x坐标
                            points[np.argmax(points[:, 1])]   # 右上：最下面的y坐标
                        ]
                    
                    board = Board()
                    board.points = [tuple(pt) for pt in sorted_points]
                    board.area = area
                    boards.append(board)
        
        # 返回面积最大的板子或空 Board 对象
        if boards:
            max_board = max(boards, key=lambda b: b.area)
            self.board = max_board
        else:
            self.board = Board()
        return self.board

    def line_intersection(self, line1, line2):
        """
        计算两条直线的交点。
        每条直线由两个点定义：[(x1, y1), (x2, y2)]。
        如果直线平行或无效，返回 None。
        """
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        # 计算分母
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # 直线平行
            return None

        # 计算交点坐标
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return (px, py)

    def get_board_center(self, board):
        """
        通过计算左上到右下和右上到左下两条直线的交点来获取板的中心。
        """
        if not board or len(board.points) != 4:
            board.center = None
            return None

        # 提取四个点
        top_left, bottom_left, bottom_right, top_right = board.points

        # 定义两条直线
        line1 = [top_left, bottom_right]  # 左上到右下
        line2 = [top_right, bottom_left]  # 右上到左下

        # 计算交点
        center = self.line_intersection(line1, line2)
        board.center = center
        return center

    def get_lazer_pos(self, tracker):
        self.lazer_center = tracker.get_laser_pixel_position()
        return self.lazer_center

    def get_circle(self, tracker):
        """
        在变换后的板子上画圆，检查当前角度点是否接近屏幕中心，若接近则递增角度并计算新点坐标。
        参数：
            frame: 输入图像
            diameter_ratio: 圆的直径与板子宽度的比例
        返回：
            self.target: 圆周上当前角度的点在原图像中的坐标
        """
        if not self.board or len(self.board.points) != 4:
            self.target = None
            return None

        # 获取板子的四个角点
        src_points = np.float32(self.board.points)
        dst_points = self.std_square

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        # 变换后板子的宽度（std_square 的宽度）
        board_width = self.std_square[1][0] - self.std_square[0][0]
        circle_radius = (board_width * self.diameter_ratio) / 2

        # 计算变换后板子中心（std_square 的中心）
        std_center_x = (self.std_square[0][0] + self.std_square[1][0]) / 2
        std_center_y = (self.std_square[0][1] + self.std_square[2][1]) / 2

        # 计算当前 circle_step 对应的圆周点
        angle_rad = math.radians(self.circle_step)
        circle_x = std_center_x + circle_radius * math.cos(angle_rad)
        circle_y = std_center_y + circle_radius * math.sin(angle_rad)

        # 将圆周上的点变换回原图像坐标
        pt = np.float32([[[circle_x, circle_y]]])
        target = cv2.perspectiveTransform(pt, M_inv)[0][0]
        self.target = tuple(target)

        if tracker.shoot == True:
            self.circle_step = (self.circle_step + 1) % 360

        return self.target

    def display(self, frame):
        """
        显示处理结果，绘制板子、中心点和目标点。
        """
        img = frame.copy()
        if self.task == 0:
            # 绘制背景板（绿色）和中心点（红色）
            if len(self.board.points) == 4:
                pts = np.array(self.board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                for i, pt in enumerate(self.board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if self.board.center is not None:
                    cv2.circle(img, (int(self.board.center[0]), int(self.board.center[1])), 5, (0, 255, 0), -1)
        
        elif self.task == 1:
            # 绘制背景板（绿色）和中心点（红色）
            if len(self.board.points) == 4:
                pts = np.array(self.board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                for i, pt in enumerate(self.board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if self.board.center is not None:
                    cv2.circle(img, (int(self.board.center[0]), int(self.board.center[1])), 5, (0, 255, 0), -1)
            # 绘制目标点（橙色）
            if self.target is not None:
                cv2.circle(img, (int(self.target[0]), int(self.target[1])), 5, (0, 165, 255), -1)
        if self.frame_center is not None:
            # 屏幕中心点（绿色）
            cv2.circle(img, (int(self.frame_center[0]), int(self.frame_center[1])), 5, (255, 0, 0), -1)
        self.result_img = img
        return img

    def run(self, frame):
        if self.task == 0:
            center = self.task1(frame)
        elif self.task == 1:
            center = self.task2(frame)
        return center
    
    def task1(self, frame):
        mask = self.process(frame)
        self.board = self.find_board(mask)
        center = self.get_board_center(self.board)
        return center

    def task2(self, frame, tracker):
        """
        检测光点和板子，使用稳定的板子进行跟踪，并计算圆周上的目标点。
        """
        mask = self.process(frame)
        self.board = self.find_board(mask)
        target = self.get_circle(tracker)  # 可调整 diameter_ratio
        return target