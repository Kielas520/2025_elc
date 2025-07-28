import numpy as np
import cv2

# 定义方形和三角形的坐标点
std_square = np.float32([[0, 0], [0, 20], [20, 20], [20, 0]])  # 方形四个顶点
std_triangle = np.float32([[5, 2], [15, 5], [10, 15]])  # 三角形三个顶点
std_star = np.float32([[10, 3], [13, 8], [19, 8], [13, 12], [17, 16], [10, 13], [3, 16], [7, 12], [1, 8], [7, 8]])

# 生成圆形的坐标点（中心(10, 10)，半径8）
theta = np.linspace(0, 2 * np.pi, 15)  # 生成15个角度，0到2π
radius = 8  # 半径
center_x, center_y = 10, 10  # 圆心
circle_x = center_x + radius * np.cos(theta)  # x坐标
circle_y = center_y + radius * np.sin(theta)  # y坐标
circle_points = np.float32(np.column_stack((circle_x, circle_y)))  # 合并为15x2的点数组

circle_points = np.int32([[18, 10], [17, 13], [15, 16], [12, 18], [8, 18], [5, 16], [3, 13], [2, 10], [3, 7], [5, 4], [8, 2], [12, 2], [15, 4], [17, 7], [18, 10]])
print(circle_points)
# 创建一个21x21的白色空白图像
img = np.ones((21, 21, 3), dtype=np.uint8) * 255

# 将坐标点转换为整数格式
square_points = np.int32(std_square)
triangle_points = np.int32(std_triangle)
circle_points = np.int32(circle_points)
star_points = np.int32(std_star)

# 绘制蓝色方形像素点 (BGR: 255, 0, 0)
for point in square_points:
    x, y = point
    img[y, x] = (255, 0, 0)  # 在(x, y)处绘制蓝色像素

# 绘制红色三角形像素点 (BGR: 0, 0, 255)
for point in triangle_points:
    x, y = point
    img[y, x] = (0, 0, 255)  # 在(x, y)处绘制红色像素

# 绘制绿色圆形像素点 (BGR: 0, 255, 0)
for point in circle_points:
    x, y = point
    img[y, x] = (0, 255, 0)  # 在(x, y)处绘制绿色像素

# 绘制绿色圆形像素点 (BGR: 0, 255, 0)
for point in star_points:
    x, y = point
    img[y, x] = (0, 0, 0)  # 在(x, y)处绘制绿色像素

# 显示图像
cv2.imshow('像素点', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('pixels.png', img)

