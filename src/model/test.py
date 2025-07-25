import cv2
import numpy as np
import time
class Board:
    def __init__(self):
        self.points = []  # 四边形角点 [左上, 左下, 右下, 右上]


class Detector:
    def __init__(self, board_min_area, bin_val=120):
        self.board_min_area = board_min_area
        self.bin_val = bin_val
        self.binary = None
        self.boards = []
        self.result_img = None
        self.draw_points = []  # 存储每个板的变换后三角形坐标
        self.std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])
        self.std_triangle = np.float32([[5, 2], [15, 5], [10, 15]])
    
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.bin_val, 255, cv2.THRESH_BINARY)
        self.binary = binary
        self.result_img = frame.copy()
        return binary
    
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
                        points[np.argmax(sum_xy)],  # 右下
                        points[np.argmin(diff_xy)]   # 右上
                    ]
                    board = Board()
                    board.points = [tuple(pt) for pt in sorted_points]
                    boards.append(board)
        
        self.boards = boards
        return boards
    
    def get_to_draw_points(self):
        """
        Computes and stores the transformed triangle coordinates for each board in self.draw_points.
        """
        self.draw_points = []
        for board in self.boards:
            if len(board.points) == 4:
                dst_pts = np.float32(board.points)
                M = cv2.getPerspectiveTransform(self.std_square, dst_pts)
                triangle_pts = cv2.perspectiveTransform(self.std_triangle.reshape(-1, 1, 2), M)
                triangle_pts = triangle_pts.reshape(-1, 2).astype(np.int32)
                self.draw_points.append([tuple(pt) for pt in triangle_pts])
        return self.draw_points
    
    def display(self):
        """
        Draws the quadrilateral and transformed triangle using self.draw_points.
        """
        img = self.result_img.copy()
        for i, board in enumerate(self.boards):
            if len(board.points) == 4:
                # 绘制四边形
                pts = np.array(board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                # 绘制三角形，使用 self.draw_points
                if i < len(self.draw_points):  # 确保索引有效
                    triangle_pts = np.array(self.draw_points[i], np.int32)
                    cv2.polylines(img, [triangle_pts], True, (0, 0, 255), 2)
                # 标记角点序号
                for j, pt in enumerate(board.points):
                    cv2.putText(img, str(j), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return img
    

    def detect(self, frame):
        start = time.time()
        binary = self.process(frame)
        print(f"Process: {time.time() - start:.3f}s")
        start = time.time()
        boards = self.find_board(binary)
        print(f"Find board: {time.time() - start:.3f}s")
        start = time.time()
        self.get_to_draw_points()
        print(f"Get draw points: {time.time() - start:.3f}s")
        return boards
def main():
    # 初始化 Detector
    detector = Detector(
        board_min_area=70000,
        bin_val=40  # 阈值，需调试
    )
    format='MJPG'
    width=640
    height=480
    fps=120
    
    # 打开摄像头
    cap = cv2.VideoCapture(4)  # 替换为视频文件路径或 0（摄像头）
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Fixed width and height order
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测
        detector.detect(frame)
        original_img = detector.display()

        # 显示
        cv2.imshow("Original", original_img)
        cv2.imshow("Binary", detector.binary)  # 调试用

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()