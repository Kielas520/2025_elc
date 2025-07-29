import math
import numpy as np
import model.detector as light

# 定义常量, 弧度转角度
RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Tracker:
    def __init__(self, img_width=640, img_height=480, f_mm=3.0, sensor_height_mm=None, T_cam_gimbal=None):
        """
        初始化Tracker类
        参数：
            img_width: 图像宽度（像素）
            img_height: 图像高度（像素）
            f_mm: 物理焦距（毫米），手动调焦后需更新
            sensor_height_mm: 传感器高度（毫米），若为None则用默认值推算
            T_cam_gimbal: 相机到云台的平移向量 [tx, ty, tz]（米），默认为 [0, 0, 0]
        """
        self.img_width = img_width
        self.img_height = img_height
        self.f_mm = f_mm
        self.T_cam_gimbal = np.array(T_cam_gimbal if T_cam_gimbal is not None else [0.0, 0.0, 0.0])
        
        # 如果未提供传感器尺寸，用默认vfov=100°推算
        if sensor_height_mm is None:
            default_vfov = 100 * DEG2RAD  # 默认100°
            self.sensor_height_mm = 2 * self.f_mm * math.tan(default_vfov / 2)
        else:
            self.sensor_height_mm = sensor_height_mm
        
        self.sensor_width_mm = self.sensor_height_mm * (img_width / img_height)
        # 计算像素焦距
        self.fx = (self.f_mm * self.img_width) / self.sensor_width_mm
        self.fy = (self.f_mm * self.img_height) / self.sensor_height_mm
        # 计算当前视场角
        self.vfov = 2 * math.atan(self.sensor_height_mm / (2 * self.f_mm)) * RAD2DEG

    def update_focal_length(self, f_mm=None, vfov=None):
        """
        更新焦距或视场角，重新计算相关参数
        参数：
            f_mm: 新物理焦距（毫米），优先使用
            vfov: 新垂直视场角（度），若f_mm为None则使用
        """
        if f_mm is not None:
            self.f_mm = f_mm
            # 重新计算vfov
            self.vfov = 2 * math.atan(self.sensor_height_mm / (2 * self.f_mm)) * RAD2DEG
        elif vfov is not None:
            self.vfov = vfov
            # 推算f_mm
            vfov_radians = vfov * DEG2RAD
            self.f_mm = self.sensor_height_mm / (2 * math.tan(vfov_radians / 2))
        
        # 更新像素焦距
        self.fx = (self.f_mm * self.img_width) / self.sensor_width_mm
        self.fy = (self.f_mm * self.img_height) / self.sensor_height_mm

    def tf_light_to_target(self, light):
        """
        转换目标点坐标，以 light.position 为原点
        """
        if light is None or light.position is None or light.target_point is None:
            return (0, 0)  # 返回默认坐标
        rel_x = light.target_point[0] - light.position[0]
        rel_y = light.target_point[1] - light.position[1]
        return (rel_x, rel_y)
    
    def tf_center_to_target(self, light):
        """
        转换目标点坐标，以图像中心为原点
        """
        center = light.target_point
        if center is None:
            return (0, 0)  # 默认坐标
        rel_x = center[0] - self.img_width / 2
        rel_y = center[1] - self.img_height / 2
        return (rel_x, rel_y)

    def pixel_to_yaw_pitch(self, center):
        """
        将像素坐标转换为偏航角和俯仰角
        参数：
            center: 像素坐标 (x, y)，图像中心为原点，x 右为正，y 下为正
        返回：
            yaw, pitch: 偏航角和俯仰角（度）
        """
        yaw = math.atan(center[0] / self.fx) * RAD2DEG
        pitch = math.atan(center[1] / self.fy) * RAD2DEG
        return yaw, pitch

    def pixel_to_gimbal_angles(self, du, dv, yaw0=0.0, pitch0=0.0, lambda_dist=None, use_translation=False):
        """
        将图像像素差值转换为云台绝对角度
        参数：
            du, dv: 图像中心为原点的像素差值 (Δu, Δv)，du 右为正，dv 下为正
            yaw0, pitch0: 云台当前角度（度）
            lambda_dist: 目标到相机的距离（米），若为None则忽略平移
            use_translation: 是否考虑平移（True为精确模式，False为远距离近似）
        返回：
            yaw_abs, pitch_abs: 云台绝对角度（度）
        """
        # 像素差值到相机坐标系方向向量，调整坐标系：du 右为正，dv 下为正
        xc = -du / self.fx  # x 轴右为正
        yc = -dv / self.fy  # y 轴下为正
        zc = 1.0

        # 云台坐标系方向向量
        if use_translation and lambda_dist is not None:
            # 精确模式：考虑平移
            xg = lambda_dist * xc + self.T_cam_gimbal[0]
            yg = lambda_dist * yc + self.T_cam_gimbal[1]
            zg = lambda_dist * zc + self.T_cam_gimbal[2]
        else:
            # 远距离近似：忽略平移
            xg, yg, zg = xc, yc, zc

        # 归一化方向向量
        norm = np.sqrt(xg**2 + yg**2 + zg**2)
        xg, yg, zg = xg / norm, yg / norm, zg / norm

        # 计算yaw和pitch（弧度）
        yaw = np.arctan2(xg, zg)
        pitch = np.arcsin(-yg)  # pitch 向上为正

        # 转换为绝对角度（度）
        yaw_abs = yaw0 + np.degrees(yaw)
        pitch_abs = pitch0 + np.degrees(pitch)

        # 限制角度范围
        yaw_abs = np.clip(yaw_abs, -180, 180)
        pitch_abs = np.clip(pitch_abs, -90, 90)

        return yaw_abs, pitch_abs

    def track(self, light, dt=1/120):
        """
        跟踪目标，融合卡尔曼滤波
        参数：
            light: 目标对象
        返回：
            yaw, pitch: 云台目标角度（度）
        """
        if light is None:
            return (0, 0)  # 如果 light 为 None，返回默认角度
        
        target = self.tf_center_to_target(light)
        yaw, pitch = self.pixel_to_gimbal_angles(target[0], target[1])
        return yaw, pitch

# 示例用法
if __name__ == "__main__":
    # 初始化 Tracker，假设初始焦距3.0mm
    tracker = Tracker(img_width=640, img_height=480, f_mm=3.0, T_cam_gimbal=[0.03, -0.09, 0.0])
    print(f"初始视场角: {tracker.vfov:.2f}°")
    
    # 测试参数
    du = 100  # 像素差值（水平，向右）
    dv = -50  # 像素差值（垂直，向上）
    yaw0 = 0  # 当前yaw（度）
    pitch0 = 0  # 当前pitch（度）
    lambda_dist = 1.0  # 目标距离1米

    # 远距离近似（忽略平移）
    yaw_abs, pitch_abs = tracker.pixel_to_gimbal_angles(du, dv, yaw0, pitch0)
    print(f"远距离近似 - Yaw: {yaw_abs:.2f}°, Pitch: {pitch_abs:.2f}°")

    # 精确模式（考虑平移）
    yaw_abs, pitch_abs = tracker.pixel_to_gimbal_angles(du, dv, yaw0, pitch0, lambda_dist=lambda_dist, use_translation=True)
    print(f"精确模式 - Yaw: {yaw_abs:.2f}°, Pitch: {pitch_abs:.2f}°")

    # 模拟手动调焦，更新焦距为3.5mm
    tracker.update_focal_length(f_mm=3.5)
    print(f"调焦后视场角: {tracker.vfov:.2f}°")
    yaw_abs, pitch_abs = tracker.pixel_to_gimbal_angles(du, dv, yaw0, pitch0)
    print(f"调焦后远距离近似 - Yaw: {yaw_abs:.2f}°, Pitch: {pitch_abs:.2f}°")