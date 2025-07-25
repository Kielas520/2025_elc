import cv2
import numpy as np

class Board:
    def __init__(self):
        self.points = []  # 四边形角点 [左上, 左下, 右下, 右上]


class Detector:
    def __init__(self, board_min_area, bin_val=120):
        self.board_min_area = board_min_area
        self.bin_val = bin_val  # 固定阈值
        self.binary = None
        self.boards = []
        self.result_img = None
        self.board_contours = None
        # 标准正方形和三角形
        self.std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])  # 左上, 左下, 右下, 右上
        self.std_triangle = np.float32([[5,2], [15, 5], [10, 15]])  # 三角形，中心 (100,100)
    
    def process(self, frame):
        # 背景板检测（灰度 + 二值化）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.bin_val, 255, cv2.THRESH_BINARY)
        self.binary = binary
        self.result_img = frame.copy()
        return binary
    
    def find_board(self, binary):
        boards = []
        self.board_contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in self.board_contours:
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
    
    #def get_to_draw_points(self):


    def display(self):
        img = self.result_img.copy()
        # 绘制背景板（四边形）和三角形
        for board in self.boards:
            if len(board.points) == 4:
                # 绘制四边形
                pts = np.array(board.points, np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                # 计算透视变换矩阵
                dst_pts = np.float32(board.points)
                M = cv2.getPerspectiveTransform(self.std_square, dst_pts)
                # 映射三角形
                triangle_pts = cv2.perspectiveTransform(self.std_triangle.reshape(-1, 1, 2), M)
                triangle_pts = triangle_pts.reshape(-1, 2).astype(np.int32)
                cv2.polylines(img, [triangle_pts], True, (0, 0, 255), 2)
                # 标记角点序号
                for i, pt in enumerate(board.points):
                    cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return img
    
    def detect(self, frame):
        binary = self.process(frame)
        boards = self.find_board(binary)
        return boards

def main():
    # 初始化 Detector
    detector = Detector(
        board_min_area=80000,
        bin_val=120  # 阈值，需调试
    )

    # 打开摄像头
    cap = cv2.VideoCapture(4)  # 替换为视频文件路径或 0（摄像头）
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

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