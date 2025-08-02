import math
import numpy as np
from model.Kalman import KalmanFilter

# 定义常量, 弧度转角度
RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Tracker:
    def __init__(self, img_width=640, img_height=480, vfov=100, use_kf=False, frame_add=20, shoot_tol=5, ref_point=(-0.03, 0, 0), offset_pitch=0.0, offset_yaw=0.0, is_mirrored=False):
        self.img_width = img_width
        self.img_height = img_height
        self.vfov = vfov
        self.is_mirrored = is_mirrored  # 是否镜像
        self.frame_add = frame_add
        self.lost = 0
        self.predict = False
        self.if_find = False
        self.if_lost = False
        self.use_kf = use_kf
        self.kf_cx = KalmanFilter()
        self.kf_cy = KalmanFilter()
        self.kf_cx.dt = 1 / 30
        self.kf_cy.dt = 1 / 30
        self.shoot_tol = shoot_tol
        self.shoot = 0
        self.ref_point = ref_point
        self.offset_pitch = offset_pitch
        self.offset_yaw = offset_yaw

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
            return (0, 0)
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

    def get_laser_pixel_position(self):
        """
        计算激光在屏幕上的像素位置
        """
        if not (isinstance(self.ref_point, (tuple, list, np.ndarray)) and len(self.ref_point) == 3):
            raise ValueError("ref_point 必须是长度为3的 (X, Y, Z) 坐标")

        laser_pos = np.array(self.ref_point, dtype=np.float32)

        yaw_rad = self.offset_yaw * DEG2RAD
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        R_y_inv = np.array([
            [cos_yaw, 0, -sin_yaw],
            [0, 1, 0],
            [sin_yaw, 0, cos_yaw]
        ], dtype=np.float32)

        pitch_rad = self.offset_pitch * DEG2RAD
        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        R_x_inv = np.array([
            [1, 0, 0],
            [0, cos_pitch, sin_pitch],
            [0, -sin_pitch, cos_pitch]
        ], dtype=np.float32)

        R_inv = R_y_inv @ R_x_inv
        laser_pos = R_inv @ laser_pos

        fov_h = self.vfov
        fov_v = self.vfov * self.img_height / self.img_width

        norm_x = laser_pos[0] / (laser_pos[2] * np.tan(np.radians(fov_h / 2))) if laser_pos[2] != 0 else 0
        norm_y = laser_pos[1] / (laser_pos[2] * np.tan(np.radians(fov_v / 2))) if laser_pos[2] != 0 else 0

        cx, cy = self.img_width / 2, self.img_height / 2
        u = cx + norm_x * (self.img_width / 2)
        v = cy + norm_y * (self.img_height / 2)

        u = np.clip(u, 0, self.img_width - 1)
        v = np.clip(v, 0, self.img_height - 1)

        if self.is_mirrored:
            u = self.img_width - 1 - u

        return (int(u), int(v))

    def calculate_relative_angles(self, pixel_point, ref_point):
        """
        计算从参考点指向图像中某个像素的相对俯仰角和偏航角，
        并判断目标点是否接近激光点（像素级判断）

        参数:
            pixel_point: 图像中的像素坐标 (u, v)
            ref_point: 相机坐标系下的参考点位置 (X_ref, Y_ref, Z_ref)

        返回:
            relative_pitch: 相对于参考点的俯仰角（度）
            relative_yaw: 相对于参考点的偏航角（度）
            arrived: 是否接近激光点（True / False）
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

        # 处理镜像图像
        if self.is_mirrored:
            u = width - 1 - u

        # 判断目标点是否接近激光点
        laser_pixel = self.get_laser_pixel_position()
        pixel_distance = np.sqrt((u - laser_pixel[0]) ** 2 + (v - laser_pixel[1]) ** 2)
        if pixel_distance < self.shoot_tol:
            arrived = True

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

        # 归一化方向向量
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm

        # 应用偏航角和俯仰角校正
        yaw_rad = -self.offset_yaw * DEG2RAD
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        R_y = np.array([
            [cos_yaw, 0, sin_yaw],
            [0, 1, 0],
            [-sin_yaw, 0, cos_yaw]
        ], dtype=np.float32)

        pitch_rad = -self.offset_pitch * DEG2RAD
        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_pitch, -sin_pitch],
            [0, sin_pitch, cos_pitch]
        ], dtype=np.float32)

        R = R_x @ R_y
        direction = R @ direction

        ref = np.array(ref_point, dtype=np.float32)
        delta = direction - ref

        relative_yaw = np.degrees(np.arctan2(delta[0], delta[2]))
        relative_pitch = np.degrees(np.arctan2(delta[1], np.sqrt(delta[0]**2 + delta[2]**2)))

        return relative_pitch, relative_yaw, arrived

    def track(self, center, dt):
        """跟踪目标并计算相对于激光的俯仰角和偏航角"""
        if center is None:
            self.lost += 1
            if self.lost <= self.frame_add:
                self.if_lost = False
                return 0, 0
            else:
                print("未检测到目标")
                self.if_lost = True
                return 0, 0
        else:
            self.lost = 0
            self.if_lost = False

        pixel_point = center
        relative_pitch, relative_yaw, arrived = self.calculate_relative_angles(pixel_point, self.ref_point)
        self.shoot = arrived

        return -relative_yaw, -relative_pitch