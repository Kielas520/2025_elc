import math
import model.detector as light


# 定义常量, 弧度转角度
RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Tracker:
    def __init__(self, img_width = 640, vfov=100, yaw_pid = 0.03, pitch_pid = 0.03):
        self.img_width = img_width
        self.vfov = vfov
        self.yaw_pid = yaw_pid
        self.pitch_pid = pitch_pid

    def tf(self, light):
        '''
        转换目标点坐标，以 light.position 为原点
        '''
        if light is None or light.position is None or light.target_point is None:
            return (0, 0)  # 返回默认坐标
        rel_x = light.target_point[0] - light.position[0]
        rel_y = light.target_point[1] - light.position[1]
        return (rel_x, rel_y)

    def pixel_to_yaw_pitch(self, center):
        """将像素坐标转换为偏航角和俯仰角"""
        vfov_radians = self.vfov * DEG2RAD
        focal_pixel_distance = (self.img_width / 2) / math.tan(vfov_radians / 2)
        if focal_pixel_distance == 0:
            focal_pixel_distance = 0.000_000_1
        yaw = math.atan(center[0] / focal_pixel_distance) * RAD2DEG
        pitch = math.atan(center[1] / focal_pixel_distance) * RAD2DEG
        return yaw, pitch

    def track(self, light):
        """跟踪目标"""
        if light is None:
            return None, None  # 如果 light 为 None，返回默认角度
        target = self.tf(light)
        yaw, pitch = self.pixel_to_yaw_pitch(target)
        return yaw, pitch