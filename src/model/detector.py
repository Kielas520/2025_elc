import cv2
import numpy as np


class Board:
    def __init__(self):
        self.points = []  # 四边形角点 [左上, 左下, 右下, 右上]
        self.area = None
        self.draw_points = []  # 存储每个板的变换后图形坐标
        self.drawn = []
        self.target_point = None
        self.center = None
        

class Detector:
    def __init__(self, board_color, board_min_area, board_max_area, separate = 20):
        self.std_square = np.float32([[0, 0], [30, 0], [30, 21], [0, 21]])
        self.std_circle = np.float32([[18, 10], [17, 13], [15, 16], [12, 18], [8, 18], [5, 16], [3, 13], [2, 10], [3, 7], [5, 4], [8, 2], [12, 2], [15, 4], [17, 7], [18, 10]])
        
        self.board_lower = board_color[0]
        self.board_upper = board_color[1]

        self.task = 1

        self.board_mask = None
        self.board_img = None  # 或初始化为空图像
        self.board_min_area = board_min_area
        self.board_max_area = board_max_area
        
        self.separate = separate

        self.board = None
        
        self.board_static = Board()  # 初始化为空 Board 对象
        self.board_current = Board()  # 当前帧板子
        self.board_prev = Board()  # 上一帧板子
        self.result_img = None

    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.board_lower, self.board_upper)
        self.board_mask = mask
        # # 背景板检测（阈值分割）
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, self.bin_min, self.bin_max, cv2.THRESH_BINARY_INV)
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

    def get_board_center(self, board):
        """
        计算板子四个角点的中心点
        参数：
            board: Board对象，包含points列表
        返回：
            center: 中心点坐标 (x, y)，若points无效则返回None
        """
        if not board or not board.points or len(board.points) != 4:
            return None
        points = np.array(board.points, dtype=np.float32)
        center = np.mean(points, axis=0)
        board.center = center
        return center

    def if_static(self):
        """
        判断板子是否稳定，通过比较四个角点的中心点距离。
        - 如果没有 static 板子，比较当前帧与上一帧。
        - 如果有 static 板子，比较 static 与当前帧，稳定则不更新 static。
        """
        # 如果当前板子无效，直接返回当前 static 板子
        if not self.board_current.points or len(self.board_current.points) != 4:
            self.board_img = None
            return self.board_static

        # 计算当前板子的中心点
        current_center = self.get_board_center(self.board_current)
        if current_center is None:
            self.board_img = None
            return self.board_static

        # 如果没有 static 板子，比较当前帧与上一帧
        if not self.board_static.points or len(self.board_static.points) != 4:
            if not self.board_prev.points or len(self.board_prev.points) != 4:
                # 上一帧也为空，将当前板子设为 static
                self.board_static = self.board_current
            else:
                # 比较当前帧与上一帧的中心点
                prev_center = self.get_board_center(self.board_prev)
                if prev_center is None:
                    self.board_static = self.board_current
                else:
                    distance = np.linalg.norm(current_center - prev_center)
                    if distance < 1:
                        self.board_static = self.board_current
            return self.board_static

        # 有 static 板子和当前板子，比较中心点
        static_center = self.get_board_center(self.board_static)
        if static_center is None:
            self.board_static = self.board_current
            return self.board_static

        distance = np.linalg.norm(current_center - static_center)

        # 如果稳定（距离 < 5），保持 self.board_static 不变
        if distance >= 5:
            # 如果不稳定，比较当前帧与上一帧
            if self.board_prev.points and len(self.board_prev.points) == 4:
                prev_center = self.get_board_center(self.board_prev)
                if prev_center is not None:
                    distance_prev = np.linalg.norm(current_center - prev_center)
                    if distance_prev < 5:
                        self.board_static = self.board_current

        return self.board_static
    def get_to_draw_points(self, board):
        """
        计算并存储板的变换后三角形坐标。
        如果 board 无效，返回空 Board。
        """
            
        draw_points = []
        if not board or len(board.points) != 4:
            board.draw_points = []
            return board
        
        # 将board.points转换为numpy数组
        src_pts = np.float32(board.points)
        M = cv2.getPerspectiveTransform(self.std_square, src_pts)
        triangle_pts = cv2.perspectiveTransform(self.std_circle.reshape(-1, 1, 2), M)
        triangle_pts = triangle_pts.reshape(-1, 2).astype(np.int32)
        
        refined_points = []
        num_points = len(triangle_pts)
        for i in range(num_points):
            p1 = triangle_pts[i]
            p2 = triangle_pts[(i + 1) % num_points]
            distance = np.linalg.norm(p1 - p2)
            
            if distance > self.separate:
                num_insert = int(distance // self.separate)
                for j in range(num_insert + 1):
                    t = j / (num_insert + 1)
                    x = int(p1[0] + t * (p2[0] - p1[0]))
                    y = int(p1[1] + t * (p2[1] - p1[1]))
                    refined_points.append((x, y))
            else:
                refined_points.append(tuple(p1))
        
        if distance <= 3 or num_insert == 0:
            refined_points.append(tuple(p2))
        
        draw_points.append(refined_points)
        board.draw_points = draw_points
        return board

    def draw(self, board, light):
        """
        绘制光点和板子，使用传入的 board（通常是 self.board_static 或 self.board_current）。
        """
        if not board or not board.draw_points or not light:
            return None
        
        light = light[0] if isinstance(light, list) else light
        
        current_triangle = board.draw_points[0]
        remaining_points = [pt for pt in current_triangle if tuple(pt) not in board.drawn]
        
        if not remaining_points:
            board.drawn.clear()
            remaining_points = current_triangle
        
        light.target_point = remaining_points[0]
        
        laser_pos = np.array(light.position, dtype=np.float32)
        target_pos = np.array(light.target_point, dtype=np.float32)
        distance = np.linalg.norm(laser_pos - target_pos)
        
        if distance < 5:
            board.drawn.append(tuple(light.target_point))
            return self.draw(board, light)
        
        self.board = board
        self.light = light
        return light

    def display(self, frame):
        """
        显示处理结果，绘制板子和光点。
        """
        img = frame.copy()
        if self.task == 1:
            # 绘制背景板（绿色）,和背景版中心点，红色
            if len(self.board.points) == 4:
                pts = np.array(self.board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                for i, pt in enumerate(self.board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if self.board.center is not None:
                    cv2.circle(img, (int(self.board.center[0]), int(self.board.center[1])), 5, (0, 255, 0), -1)
        
        elif self.task ==2:
            # 绘制背景板（绿色）,和背景版中心点，红色
            if len(self.board.points) == 4:
                pts = np.array(self.board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                for i, pt in enumerate(self.board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 绘制目标（红色）
            if self.board.draw_points and len(self.board.draw_points) > 0:
                triangle = self.board.draw_points[0]
                pts = np.array(triangle, np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            
            # 绘制待绘制点（蓝色）
            if self.board.draw_points and len(self.board.draw_points) > 0:
                current_triangle = self.board.draw_points[0]
                for point in current_triangle:
                    if tuple(point) not in self.board.drawn:
                        cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            
            # 绘制已处理的点（白色）
            for point in self.board.drawn:
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)
        
        self.result_img = img
        return img


    def task1(self, frame):
        mask = self.process(frame)
        self.board = self.find_board(mask)
        center = self.get_board_center(self.board)
        return center


    def task2(self, frame):
        """
        检测光点和板子，使用稳定的板子进行跟踪。
        """
        # 处理帧，生成掩膜和二值图像
        mask = self.process(frame)
        
        # 检测当前板子并存储到 self.board_current
        self.board_current = self.find_board(mask)
        
        # 判断板子是否稳定，更新 self.board_static
        static_board = self.if_static()
        
        # 使用静态板子（如果存在）进行跟踪，否则使用当前板子
        board_to_use = static_board if static_board.points else self.board_current
        
        # 计算要绘制的点
        board_to_use = self.get_to_draw_points(board_to_use)
        
        # 绘制光点和板子
        light = self.draw(board_to_use, light)
        
        # 更新 self.board 为显示用
        self.board = board_to_use
        
        # 更新上一帧板子
        self.board_prev = self.board_current
        
        return light