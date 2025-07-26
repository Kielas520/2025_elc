import cv2
import numpy as np

class Light:
    def __init__(self):
        self.position = None
        self.area = None

class Board:
    def __init__(self):
        self.points = []  # 四边形角点 [左上, 左下, 右下, 右上]

class Detector:
    def __init__(self, color, light_min_area, board_min_area, bin_val, canny_min, canny_max, kernel_x, kernel_y):
        self.bgr_upper = color[0]
        self.bgr_lower = color[1]
        self.light_min_area = light_min_area
        self.mask = None
        self.lights = []

        self.canny_min = canny_min
        self.canny_max = canny_max

        self.kernel_x = kernel_x
        self.kernel_y = kernel_y

        self.binary = None
        self.bin_val = bin_val
        self.board_min_area = board_min_area
        self.boards = []

        self.draw_points = []  # 存储每个板的变换后图形坐标
        self.std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])
        self.std_triangle = np.float32([[5, 2], [15, 5], [10, 15]])
        self.drawn = []
        self.result_img = None
    
    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.bgr_lower, self.bgr_upper)
        self.mask = mask

        # 背景板检测（Canny 边缘检测）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 中值模糊减少噪声
        #gray = cv2.medianBlur(gray, 3)
        # 阈值分割，检测黑色区域（灰度值 < 50）
        _, binary = cv2.threshold(gray, self.canny_min, self.canny_max, cv2.THRESH_BINARY_INV)
        kernel = (self.kernel_x, self.kernel_y)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 可选：轻微模糊以减少噪声
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny 边缘检测
        # edges = cv2.Canny(gray, self.canny_min, self.canny_max, apertureSize=3)
        # 可选：膨胀操作连接断续边缘
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_x, self.kernel_y))
        # 腐蚀
        #binary = cv2.erode(binary, kernel, iterations=1)
        #binary = cv2.dilate(binary, kernel, iterations=1)
        self.binary = binary

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

    def get_to_draw_points(self, boards):
        """
        Computes and stores the transformed triangle coordinates for each board in self.draw_points.
        If the distance between adjacent points exceeds 80 pixels, insert additional points.
        """
        draw_points = []
        for board in boards:
            if len(board.points) == 4:
                # 进行透视变换生成三角形点
                dst_pts = np.float32(board.points)
                M = cv2.getPerspectiveTransform(self.std_square, dst_pts)
                triangle_pts = cv2.perspectiveTransform(self.std_triangle.reshape(-1, 1, 2), M)
                triangle_pts = triangle_pts.reshape(-1, 2).astype(np.int32)
                
                # 处理三角形点，插入额外点
                refined_points = []
                num_points = len(triangle_pts)
                for i in range(num_points):
                    p1 = triangle_pts[i]
                    p2 = triangle_pts[(i + 1) % num_points]  # 下一个点，闭合循环
                    distance = np.linalg.norm(p1 - p2)
                    
                    # 计算需要插入的点数
                    if distance > 80:
                        num_insert = int(distance // 80)  # 向下取整
                        for j in range(num_insert + 1):  # 包含起点
                            t = j / (num_insert + 1)  # 插值比例
                            x = int(p1[0] + t * (p2[0] - p1[0]))
                            y = int(p1[1] + t * (p2[1] - p1[1]))
                            refined_points.append((x, y))
                    else:
                        refined_points.append(tuple(p1))
                
                # 添加最后一个点以确保闭合（如果没有插入点）
                if distance <= 80 or num_insert == 0:
                    refined_points.append(tuple(p2))
                
                draw_points.append(refined_points)
        
        self.draw_points = draw_points
        return draw_points

    def draw(self, draw_points, lights):
        # 找到激光点（面积最小的光点）
        dot = min(lights, key=lambda light: light.area) if lights else None
        
        # 如果没有绘制点，返回原点
        if not draw_points:
            return (0, 0)
        
        # 获取当前需要追踪的点
        current_triangle = draw_points[0]  # 始终追踪第零个三角形
        remaining_points = [pt for pt in current_triangle if tuple(pt) not in self.drawn]
        
        # 如果所有点都已绘制完成，清空 self.drawn 以重复绘制
        if not remaining_points:
            self.drawn.clear()
            remaining_points = current_triangle
        
        # 选择当前追踪的点（remaining_points 中的第一个点）
        target_point = remaining_points[0]
        
        # 如果没有激光点，返回第零个三角形的第一个点
        if not dot:
            return tuple(target_point)
        
        # 计算激光点与目标点的距离
        laser_pos = np.array(dot.position, dtype=np.float32)
        target_pos = np.array(target_point, dtype=np.float32)
        distance = np.linalg.norm(laser_pos - target_pos)
        
        # 判断是否重合（距离阈值设为 10 像素，可调整）
        if distance < 20:
            # 激光点与目标点重合，添加到已处理列表
            self.drawn.append(tuple(target_point))
            # 递归调用以处理下一个点
            return self.draw(draw_points, lights)
        
        # 返回当前需要瞄准的点的坐标
        return tuple(target_point)

    def tf_point(self, point, frame):
        '''
        转换坐标原点，让原点变成图像中心位置
        '''
        if frame is None:
            raise ValueError("No frame available for coordinate transformation")
        
        height, width = frame.shape[:2]
        center_x = point[0] - width / 2
        center_y = point[1] - height / 2
        return (center_x, center_y)

    def display(self, frame):
        img = frame.copy()  # Create a copy for drawing
        
        # 绘制背景板（四边形连线，绿色）
        for board in self.boards:
            if len(board.points) == 4:
                pts = np.array(board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                # 标记角点序号
                for i, pt in enumerate(board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制目标三角形（红色）
        if self.draw_points:
            triangle = self.draw_points[0]  # 取第零个三角形
            pts = np.array(triangle, np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)  # 红色线条
        
        # 绘制待绘制点（蓝色）
        if self.draw_points:
            current_triangle = self.draw_points[0]
            for point in current_triangle:
                if tuple(point) not in self.drawn:
                    cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # 蓝色
        
        # 绘制激光点（绿色）
        for light in self.lights:
            if light.position:
                cv2.circle(img, (int(light.position[0]), int(light.position[1])), 5, (0, 255, 0), -1)
        
        # 绘制已处理的点（白色）
        for point in self.drawn:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)
        
        self.result_img = img
        return img
    
    def detect(self, frame):
        mask, binary = self.process(frame)
        lights = self.find_light(mask)
        boards = self.find_board(binary)
        draw_points = self.get_to_draw_points(boards)
        point = self.draw(draw_points, lights)
        point = self.tf_point(point, frame)
        return point