import cv2
import numpy as np
from model.template import ORBMatcher

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
        

class Detector:
    def __init__(self, color, light_min_area, board_min_area, board_max_area, bin_min, bin_max, kernel_x, kernel_y, shrink_distance = 25, min_pic_area = 150, max_pic_area = 2300):
        self.std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])
        self.std_triangle = np.float32([[15, 10], [8, 14], [8, 6]])
        self.std_circle = np.float32([[18, 10], [17, 13], [15, 16], [12, 18], [8, 18], [5, 16], [3, 13], [2, 10], [3, 7], [5, 4], [8, 2], [12, 2], [15, 4], [17, 7], [18, 10]])
        self.std_insquare = np.float32([[4, 4], [4, 16], [16, 16], [16, 4]])
        self.std_star = np.float32([[10, 3], [13, 8], [19, 8], [13, 12], [17, 16], [10, 13], [3, 16], [7, 12], [1, 8], [7, 8]])

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
        self.shrink_distance = shrink_distance
        self.board_img = None  # 或初始化为空图像
        self.board_min_area = board_min_area
        self.board_max_area = board_max_area
        
        # 定义面积上下阈值
        self.min_pic_area = min_pic_area  # 最小面积阈值
        self.max_pic_area = max_pic_area  # 最大面积阈值，可根据需要调整

        self.board = None
        self.board_shape = self.std_star
        
        self.board_static = Board()  # 初始化为空 Board 对象
        self.board_current = Board()  # 当前帧板子
        self.board_prev = Board()  # 上一帧板子
        
        self.result_img = None
        self.sift_matcher = ORBMatcher(min_match_count=5, ratio_thresh=0.75)
        # 添加多个模板
        self.sift_matcher.add_template("triangle", "src/pic/tri.jpg")
        self.sift_matcher.add_template("rectangle", "src/pic/rect.jpg")
        self.sift_matcher.add_template("circle", "src/pic/circle.jpg")
        self.sift_matcher.add_template("None", "src/pic/board.jpg")

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
                        # 新的排序规则：左上最左，左下最上，右下最右，右上最下
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
        
        # 返回面积最大的光点或空 Light 对象
        if lights:
            max_light = max(lights, key=lambda l: l.area)
            self.light = max_light
        else:
            self.light = Light()
        return self.light

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
                self.board_img = self.get_board_img(self.board_current)
                self.board_static = self.board_current
            else:
                # 比较当前帧与上一帧的中心点
                prev_center = self.get_board_center(self.board_prev)
                if prev_center is None:
                    self.board_img = self.get_board_img(self.board_current)
                    self.board_static = self.board_current
                else:
                    distance = np.linalg.norm(current_center - prev_center)
                    if distance < 1:
                        self.board_img = self.get_board_img(self.board_current)
                        self.board_static = self.board_current
            return self.board_static

        # 有 static 板子和当前板子，比较中心点
        static_center = self.get_board_center(self.board_static)
        if static_center is None:
            self.board_img = self.get_board_img(self.board_current)
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
                        self.board_img = self.get_board_img(self.board_current)
                        self.board_static = self.board_current

        return self.board_static


    def get_board_img(self, board):
        # 将board.points转换为numpy数组
        src_pts = np.float32(board.points)
        # 向内缩小四边形5像素
        shrunk_pts = []
        num_points = len(src_pts)
        
        for i in range(num_points):
            # 获取当前点和相邻点
            p = src_pts[i]
            prev_p = src_pts[(i - 1) % num_points]
            next_p = src_pts[(i + 1) % num_points]
            
            # 计算两条边的向量
            vec1 = p - prev_p  # 从前一个点到当前点
            vec2 = next_p - p  # 从当前点到下一个点
            
            # 计算法向量（逆时针旋转90度）
            normal1 = np.array([-vec1[1], vec1[0]])
            normal2 = np.array([-vec2[1], vec2[0]])
            
            # 归一化法向量
            normal1 = normal1 / (np.linalg.norm(normal1) + 1e-6)
            normal2 = normal2 / (np.linalg.norm(normal2) + 1e-6)
            
            # 计算平均内向法向量
            avg_normal = (normal1 + normal2) / 2
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-6)
            
            # 向内移动点
            shrunk_point = p + avg_normal * self.shrink_distance
            shrunk_pts.append(shrunk_point)
        
        # 将缩小的点转换为numpy数组
        src_pts = np.float32(shrunk_pts)
        # 定义目标矩形（输出ROI的大小，可以自定义）
        roi_width, roi_height = 50, 50  # 可根据需求调整
        dst_pts = np.float32([[0, 0], [roi_width, 0], [roi_width, roi_height], [0, roi_height]])
        # 计算透视变换矩阵（从目标矩形到四边形）
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # 应用透视变换到self.binary，提取四边形ROI并变换为矩形
        board_img = cv2.warpPerspective(self.binary, M, (roi_width, roi_height))
        
        self.get_shape(board_img)
        return board_img

    def get_shape(self, board_img):
        # 执行匹配
        result = self.sift_matcher.match(board_img, show_result=False)
        if result:
            sides = result['name']
            if sides == 'triangle':
                self.board_shape = self.std_triangle

            elif sides == 'rectangle':
                self.board_shape = self.std_insquare

            elif sides == 'circle':
                self.board_shape = self.std_circle

            elif sides == 'None':
                pass

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
        triangle_pts = cv2.perspectiveTransform(self.board_shape.reshape(-1, 1, 2), M)
        triangle_pts = triangle_pts.reshape(-1, 2).astype(np.int32)
        
        # 处理三角形点，插入额外点
        refined_points = []
        num_points = len(triangle_pts)
        for i in range(num_points):
            p1 = triangle_pts[i]
            p2 = triangle_pts[(i + 1) % num_points]
            distance = np.linalg.norm(p1 - p2)
            
            if distance > 10:
                num_insert = int(distance // 10)
                for j in range(num_insert + 1):
                    t = j / (num_insert + 1)
                    x = int(p1[0] + t * (p2[0] - p1[0]))
                    y = int(p1[1] + t * (p2[1] - p1[1]))
                    refined_points.append((x, y))
            else:
                refined_points.append(tuple(p1))
        
        if distance <= 5 or num_insert == 0:
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
        
        # 绘制背景板（绿色）
        if len(self.board.points) == 4:
            pts = np.array(self.board.points, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(self.board.points):
                cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制目标三角形（红色）
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
        
        # 绘制激光点（绿色）
        if self.light and self.light.position:
            cv2.circle(img, (int(self.light.position[0]), int(self.light.position[1])), 5, (0, 255, 0), -1)
        
        # 绘制已处理的点（白色）
        for point in self.board.drawn:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)
        
        self.result_img = img
        return img

    def detect(self, frame):
        """
        检测光点和板子，使用稳定的板子进行跟踪。
        """
        # 处理帧，生成掩膜和二值图像
        mask, binary = self.process(frame)
        
        # 检测光点
        light = self.find_light(mask)
        
        # 检测当前板子并存储到 self.board_current
        self.board_current = self.find_board(binary)
        
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