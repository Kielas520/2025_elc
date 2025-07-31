import math
from model.Kalman import KalmanFilter 
import numpy as np

# 定义常量, 弧度转角度
RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Tracker:
    def __init__(self, img_width = 640, img_height = 480, vfov = 100, yaw_pid = 0.003, pitch_pid = 0.003, use_kf = False, frame_add = 20, shoot_tol = 5, ref_point = (0.04, 0, 0)):
        self.img_width = img_width
        self.img_height = img_height
        self.vfov = vfov

        self.frame_add = frame_add  # 补帧数
        self.lost = 0  # 丢失帧计数
        self.predict = False  # 是否处于预测状态
        self.if_find = False  # 是否找到目标

        self.use_kf = use_kf  # 是否使用卡尔曼滤波
        # 初始化卡尔曼滤波器
        self.kf_cx = KalmanFilter()  # x 坐标滤波器
        self.kf_cy = KalmanFilter()  # y 坐标滤波器
        
        self.kf_cx.dt = 1 / 30
        self.kf_cy.dt = 1 / 30

        self.yaw_pid = yaw_pid
        self.pitch_pid = pitch_pid
        self.shoot_tol = shoot_tol
        self.shoot = 0
        
        self.ref_point = ref_point  # 单位：米, 激光相对于相机的位置（相机在激光左侧4厘米）

    def update_dt(self, dt):
        """更新卡尔曼滤波器时间步长"""
        self.kf_cx.dt = dt
        self.kf_cy.dt = dt

    def kf_predict(self):
        """执行卡尔曼滤波预测"""
        self.kf_cx.predict()
        self.kf_cy.predict()

    def get_kf_state(self):
        """获取卡尔曼滤波器当前状态"""
        return (self.kf_cx.get_state(), self.kf_cy.get_state())

    def reset_kf(self):
        """重置卡尔曼滤波器"""
        self.kf_cx.reset()
        self.kf_cy.reset()

    def kf_update(self, center):
        """更新卡尔曼滤波器状态"""
        self.kf_cx.update(center[0])
        self.kf_cy.update(center[1])

    def tf_center_to_target(self, center):
        """
        转换目标点坐标，以图像中心为原点
        """
        if center is None:
            return (0, 0)  # 默认坐标
        rel_x = center[0] - self.img_width / 2
        rel_y = center[1] - self.img_height / 2
        return (-rel_x, rel_y)

    def pixel_to_yaw_pitch(self, center):
        """将像素坐标转换为偏航角和俯仰角"""
        vfov_radians = self.vfov * DEG2RAD
        focal_pixel_distance = (self.img_width / 2) / math.tan(vfov_radians / 2)
        if focal_pixel_distance == 0:
            focal_pixel_distance = 0.000_000_1
        yaw = math.atan(center[0] / focal_pixel_distance) * RAD2DEG
        pitch = math.atan(center[1] / focal_pixel_distance) * RAD2DEG
        return yaw, pitch

    def calculate_relative_angles(self, pixel_point, ref_point, pixel_threshold=10):
        """
        计算从参考点指向图像中某个像素的相对俯仰角和偏航角，
        并判断该像素是否靠近画面中心（像素级判断）

        参数:
            pixel_point: 图像中的像素坐标 (u, v)
            ref_point: 相机坐标系下的参考点位置 (X_ref, Y_ref, Z_ref)
            pixel_threshold: 到画面中心的像素距离阈值

        返回:
            relative_pitch: 相对于参考点的俯仰角（度）
            relative_yaw: 相对于参考点的偏航角（度）
            arrived: 是否接近画面中心（True / False）
        """
        relative_pitch = 0
        relative_yaw = 0
        arrived = False
    
        # 参数检查
        if not (isinstance(pixel_point, (tuple, list)) and len(pixel_point) == 2):
            raise ValueError("pixel_point 必须是长度为2的 (u, v) 元组或列表")
        if not isinstance(self.vfov, (int, float)):
            raise ValueError("vfov 必须是 float 或 int")
        if not (isinstance(ref_point, (tuple, list, np.ndarray)) and len(ref_point) == 3):
            raise ValueError("ref_point 必须是长度为3的 (X, Y, Z) 坐标")

        u, v = pixel_point
        width, height = self.img_width, self.img_height
        cx, cy = width / 2, height / 2

        # 判断像素点是否接近画面中心
        pixel_distance = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
        arrived = pixel_distance < pixel_threshold

        # 归一化图像坐标（范围：[-1, 1]）
        norm_x = (u - cx) / (width / 2)
        norm_y = (v - cy) / (height / 2)

        # 估算垂直视角
        fov_h = self.vfov
        fov_v = self.vfov * height / width

        # 像素方向向量（相机坐标系）
        x_dir = np.tan(np.radians(fov_h / 2)) * norm_x
        y_dir = np.tan(np.radians(fov_v / 2)) * norm_y
        z_dir = 1.0
        direction = np.array([x_dir, y_dir, z_dir], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm

        ref = np.array(ref_point, dtype=np.float32)
        delta = direction - ref

        # 计算角度
        relative_yaw = np.degrees(np.arctan2(delta[0], delta[2]))
        relative_pitch = np.degrees(np.arctan2(delta[1], np.sqrt(delta[0]**2 + delta[2]**2)))

        return relative_pitch, relative_yaw, arrived

    def track(self, center, dt):
        """跟踪目标并计算相对于激光的俯仰角和偏航角"""
        if center is None:
            # 没有检测到目标
            if self.use_kf:
                self.lost += 1
                if self.lost <= self.frame_add and self.predict:
                    self.update_dt(dt)  # 更新时间步长
                    self.kf_predict()  # 预测下一步
                    center = self.get_kf_state()  # 获取预测的中心点
                    self.if_find = True
                else:
                    print("未检测到目标")
                    self.reset_kf()  # 重置滤波器
                    self.lost = 0
                    self.predict = False
                    self.if_find = False
                    return None, None
            else:
                print("未检测到目标")
                self.if_find = False
                return None, None
        else:
            # 检测到目标
            self.predict = True
            self.if_find = True
            self.lost = 0
            if self.use_kf:
                self.update_dt(dt)  # 更新时间步长
                self.kf_update(center)  # 更新滤波器
                self.kf_predict()  # 预测下一步
                center = self.get_kf_state()  # 获取滤波后的中心点

        # 直接使用 center 作为像素坐标 (u, v)
        pixel_point = center
        # 计算相对于激光的俯仰角和偏航角
        relative_pitch, relative_yaw, arrived = self.calculate_relative_angles(pixel_point, self.ref_point, self.shoot_tol)

        # 使用 arrived 判断是否触发射击
        self.shoot = arrived

        return relative_yaw, relative_pitch