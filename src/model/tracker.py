import math
import model.detector as Blobs
from model.Kalman import KalmanFilter 

# 定义常量, 弧度转角度
RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Tracker:
    def __init__(self, img_width=1920, vfov=80, use_kf = True, frame_add = 35):
        self.img_width = img_width
        self.vfov = vfov
        self.use_kf = use_kf  # 是否使用卡尔曼滤波
        self.frame_add = frame_add  # 补帧数
        self.lost = 0  # 丢失帧计数
        self.predict = False  # 是否处于预测状态
        self.if_find = False  # 是否找到目标
        # 初始化卡尔曼滤波器
        self.kf_cx = KalmanFilter()  # x 坐标滤波器
        self.kf_cy = KalmanFilter()  # y 坐标滤波器

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

    def select_target(self, blobs):
        """选择目标（基于面积最大的 blob）"""
        if len(blobs) == 0:
            return None
        else:
            blob = max(blobs, key=lambda blob: blob.area)
            return blob.center

    def pixel_to_yaw_pitch(self, center):
        """将像素坐标转换为偏航角和俯仰角"""
        vfov_radians = self.vfov * DEG2RAD
        focal_pixel_distance = (self.img_width / 2) / math.tan(vfov_radians / 2)
        if focal_pixel_distance == 0:
            focal_pixel_distance = 0.000_000_1
        yaw = math.atan(center[0] / focal_pixel_distance) * RAD2DEG
        pitch = math.atan(center[1] / focal_pixel_distance) * RAD2DEG
        return yaw, pitch

    def track(self, point, dt=1/120):
        """跟踪目标，融合卡尔曼滤波"""
        center = point
        
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
                    return 0, 0
            else:
                print("未检测到目标")
                self.if_find = False
                return 0, 0
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
            
        yaw, pitch = self.pixel_to_yaw_pitch(center)
        return yaw, pitch