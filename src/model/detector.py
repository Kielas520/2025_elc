import cv2
import numpy as np

class Light:
    def __init__(self):
        self.position = None
        self.target_point = None
        self.area = None

class Board:
    def __init__(self):
        self.points = []  # 四边形角点 [左上, 左下, 右下, 右上]
        self.area = None
        self.draw_points = []  # 存储每个板的变换后图形坐标
        self.drawn = []
        self.shape_type = None  # 新增：存储板子形状（"circle", "quad", "triangle", "draw")(3,2,1,0)

class Detector:
    def __init__(self, color, light_min_area, board_min_area, bin_min, bin_max, kernel_x, kernel_y):
        self.bgr_upper = color[0]
        self.bgr_lower = color[1]
        self.light_min_area = light_min_area
        self.mask = None
        self.light = None
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.binary = None
        self.board_min_area = board_min_area
        self.board = None
        self.board_static = Board()  # 初始化为空 Board 对象
        self.board_current = Board()  # 当前帧板子
        self.board_prev = Board()  # 上一帧板子
        self.std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])
        self.std_triangle = np.float32([[15, 10], [8, 14], [8, 6]])
        self.std_circle = np.float32([[18, 10], [17, 13], [15, 16], [12, 18], [8, 18], [5, 16], [3, 13], [2, 10], [3, 7], [5, 4], [8, 2], [12, 2], [15, 4], [17, 7], [18, 10]])
        self.std_insquare = np.float32([[4, 4], [4, 16], [16, 16], [16, 4]])
        self.result_img = None
        self.shape_type = 0
        self.shape_to_use = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])
    
    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.bgr_lower, self.bgr_upper)
        kernel = np.ones((self.kernel_x, self.kernel_y), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.mask = mask

        # 背景板检测（阈值分割）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.bin_min, self.bin_max, cv2.THRESH_BINARY_INV)
        self.binary = binary

        return mask, binary

    def find_board(self, binary):
        boards = []
        board_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in board_contours:
            area = cv2.contourArea(contour)
            if area > self.board_min_area:
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
        
        if boards:
            max_board = max(boards, key=lambda b: b.area)
            self.board = max_board
        else:
            self.board = Board()
        return self.board

    def find_light(self, mask):
        lights = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.light_min_area:
                x, y, w, h = cv2.boundingRect(contour)
                light = Light()
                cx = x + w//2
                cy = y + h //2
                light.position = (cx, cy)
                light.area = area
                lights.append(light)
        
        if lights:
            max_light = max(lights, key=lambda l: l.area)
            self.light = max_light
        else:
            self.light = Light()
        return self.light

    def detect_shape(self, board):
        """
        检测板子内的形状：圆形、四边形、三角形或画板。
        """
        if not board or not board.points or len(board.points) != 4:
            self.shape_type = 0  # 默认画板
            return board

        # 获取板子区域的角点并向内缩小30像素
        points = np.array(board.points, dtype=np.float32)
        
        # 计算板子的几何中心
        center = np.mean(points, axis=0)
        
        # 向内缩小30像素
        shrink_distance = 10  # 缩小距离（像素）
        shrunk_points = []
        for point in points:
            # 计算当前点到中心的向量
            vector = point - center
            # 计算向量的长度
            length = np.linalg.norm(vector)
            if length > 0:  # 避免除以0
                # 按比例向内移动，保持方向不变
                scale = (length - shrink_distance) / length if length > shrink_distance else 0
                shrunk_point = center + scale * vector
                shrunk_points.append(shrunk_point)
            else:
                # 如果点在中心（极少见），直接使用原点
                shrunk_points.append(point)
        
        # 转换为浮点型数组
        shrunk_points = np.array(shrunk_points, dtype=np.float32)
        
        # 如果缩小后点无效（例如重合或面积太小），仍按画板处理
        if len(set(tuple(pt) for pt in shrunk_points)) < 4:
            board.shape_type = 0
            self.shape_type = 0
            return board

        # 进行透视变换
        width, height = 100, 100  # std_square 的大小
        M = cv2.getPerspectiveTransform(shrunk_points, self.std_square)
        # 提取二值化图像的 ROI 并进行透视变换
        roi = cv2.warpPerspective(self.binary, M, (width, height))
        # 查找轮廓
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = min(contours, key=cv2.contourArea)
        # 拟合多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 5:
            board.shape_type = 3
            self.shape_type = 3
            return board

        elif len(approx) == 4:
            board.shape_type = 2
            self.shape_type = 2
        elif len(approx) == 3:
            board.shape_type = 1
            self.shape_type = 1
        else:
            board.shape_type = 0
            self.shape_type = 0
        
        return board

    def if_static(self):
        """
        判断板子是否稳定，并在稳定时进行形状检测。
        """
        if not self.board_current.points:
            return self.board_static

        if not self.board_static.points:
            if not self.board_prev.points:
                self.board_static = self.board_current
            else:
                current_top_left = np.array(self.board_current.points[0], dtype=np.float32)
                prev_top_left = np.array(self.board_prev.points[0], dtype=np.float32)
                distance = np.linalg.norm(current_top_left - prev_top_left)
                if distance < 5:
                    self.board_static = self.board_current
                    # 板子稳定后进行形状检测
                    self.board_static = self.detect_shape(self.board_static)
            return self.board_static

        current_top_left = np.array(self.board_current.points[0], dtype=np.float32)
        static_top_left = np.array(self.board_static.points[0], dtype=np.float32)
        distance = np.linalg.norm(current_top_left - static_top_left)

        if distance >= 5:
            if self.board_prev.points:
                prev_top_left = np.array(self.board_prev.points[0], dtype=np.float32)
                distance_prev = np.linalg.norm(current_top_left - prev_top_left)
                if distance_prev < 5:
                    self.board_static = self.board_current
                    # 板子稳定后进行形状检测
                    self.board_static = self.detect_shape(self.board_static)

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

        dst_pts = np.float32(board.points)
        if self.shape_type == 1:
            self.shape_to_use = self.std_triangle
        elif self.shape_type == 2:
            self.shape_to_use = self.std_insquare
        elif self.shape_type == 3:
            self.shape_to_use = self.std_circle
        M = cv2.getPerspectiveTransform(self.std_square, dst_pts)
        triangle_pts = cv2.perspectiveTransform(self.shape_to_use.reshape(-1, 1, 2), M)
        triangle_pts = triangle_pts.reshape(-1, 2).astype(np.int32)
        
        refined_points = []
        num_points = len(triangle_pts)
        for i in range(num_points):
            p1 = triangle_pts[i]
            p2 = triangle_pts[(i + 1) % num_points]
            distance = np.linalg.norm(p1 - p2)
            
            if distance > 20:
                num_insert = int(distance // 20)
                for j in range(num_insert + 1):
                    t = j / (num_insert + 1)
                    x = int(p1[0] + t * (p2[0] - p1[0]))
                    y = int(p1[1] + t * (p2[1] - p1[1]))
                    refined_points.append((x, y))
            else:
                refined_points.append(tuple(p1))
        
        if distance <= 80 or num_insert == 0:
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
        显示处理结果，绘制板子和光点，并显示板子形状。
        """
        img = frame.copy()
        
        # 绘制背景板（绿色）
        if len(self.board.points) == 4:
            pts = np.array(self.board.points, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(self.board.points):
                cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if self.shape_type > 0:
            # 绘制目标
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
        
        # 绘制激光点（绿色）
        if self.light and self.light.position:
            cv2.circle(img, (int(self.light.position[0]), int(self.light.position[1])), 5, (0, 255, 0), -1)
        
        self.result_img = img
        return img

    def detect(self, frame):
        """
        检测光点和板子，使用稳定的板子进行跟踪。
        """
        mask, binary = self.process(frame)
        self.board_current = self.find_board(binary)
        light = self.find_light(mask)
        static_board = self.if_static()
        board_to_use = static_board if static_board.points else self.board_current
        board_to_use = self.get_to_draw_points(board_to_use)
        light = self.draw(board_to_use, light)
        self.board = board_to_use
        self.board_prev = self.board_current
        return light