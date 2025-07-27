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
        self.board_static = None

        self.std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])
        self.std_triangle = np.float32([[5, 2], [15, 5], [10, 15]])

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
        _, binary = cv2.threshold(gray, self.bin_min, self.bin_max, cv2.THRESH_BINARY_INV)
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
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    points = approx.reshape(4, 2)
                    sum_xy = points.sum(axis=1)
                    diff_xy = points[:, 0] - points[:, 1]
                    sorted_points = [
                        points[np.argmin(sum_xy)],  # 左上
                        points[np.argmax(diff_xy)],  # 左下
                        points[np.argmax(sum_xy)],   # 右下
                        points[np.argmin(diff_xy)]   # 右上
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
            self.board = Board()  # 空 Board 对象
        return self.board

    def find_light(self, mask):
        lights = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.light_min_area:
                x, y, _, _ = cv2.boundingRect(contour)
                light = Light()
                light.position = (x, y)
                light.area = area  # 设置 area 属性
                lights.append(light)
        
        # 返回面积最大的光点或空 Light 对象
        if lights:
            max_light = max(lights, key=lambda l: l.area)
            self.light = max_light  # 存储单一最大光点
        else:
            self.light = Light()  # 空 Light 对象
        return self.light

    def get_to_draw_points(self, board):
        """
        Computes and stores the transformed triangle coordinates for the board.
        Returns the board or an empty Board if invalid.
        """
        draw_points = []
        if not board or len(board.points) != 4:  # 检查 board 是否有效
            board.draw_points = []
            return board
        
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
            
            if distance > 80:
                num_insert = int(distance // 80)
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
        if not board or not board.draw_points or not light:  # 检查 board 和 light 是否有效
            return None
        
        light = light[0] if isinstance(light, list) else light  # 确保 light 是单个 Light 对象
        
        # 获取当前需要追踪的点
        current_triangle = board.draw_points[0]  # 始终追踪第零个三角形
        remaining_points = [pt for pt in current_triangle if tuple(pt) not in board.drawn]
        
        # 如果所有点都已绘制完成，清空 drawn 以重复绘制
        if not remaining_points:
            board.drawn.clear()
            remaining_points = current_triangle
        
        # 选择当前追踪的点
        light.target_point = remaining_points[0]
        
        # 计算激光点与目标点的距离
        laser_pos = np.array(light.position, dtype=np.float32)
        target_pos = np.array(light.target_point, dtype=np.float32)
        distance = np.linalg.norm(laser_pos - target_pos)
        
        # 判断是否重合（距离阈值设为 20 像素）
        if distance < 20:
            board.drawn.append(tuple(light.target_point))
            return self.draw(board, light)  # 递归调用处理下一个点
        
        self.board = board
        self.light = light
        return light

    # def tf_point(self, point, frame):
    #     '''
    #     转换坐标原点，让原点变成图像中心位置
    #     '''
    #     if frame is None:
    #         raise ValueError("No frame available for coordinate transformation")
        
    #     height, width = frame.shape[:2]
    #     center_x = point[0] - width / 2
    #     center_y = point[1] - height / 2
    #     return (center_x, center_y)

    def display(self, frame):
        img = frame.copy()  # Create a copy for drawing
        
        # 绘制背景板（四边形连线，绿色）
        if len(self.board.points) == 4:
            pts = np.array(self.board.points, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            # 标记角点序号
            for i, pt in enumerate(self.board.points):
                cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制目标三角形（红色）
        if self.board.draw_points and len(self.board.draw_points) > 0:
            triangle = self.board.draw_points[0]  # 取第零个三角形的点列表
            pts = np.array(triangle, np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)  # 红色线条

        # 绘制待绘制点（蓝色）
        if self.board.draw_points and len(self.board.draw_points) > 0:
            current_triangle = self.board.draw_points[0]  # 取第零个三角形的点列表
            for point in current_triangle:
                if tuple(point) not in self.board.drawn:
                    cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # 蓝色
        
        # 绘制激光点（绿色）
        if self.light.position:
            cv2.circle(img, (int(self.light.position[0]), int(self.light.position[1])), 5, (0, 255, 0), -1)
        
        # 绘制已处理的点（白色）
        for point in self.board.drawn:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)
        
        self.result_img = img
        return img
    
    def if_static(self):
        self.board


    def detect(self, frame):
        mask, binary = self.process(frame)
        light = self.find_light(mask)
        static = self.if_static()
        board = self.find_board(binary, static)
        board = self.get_to_draw_points(board)
        light = self.draw(board, light)
        return light