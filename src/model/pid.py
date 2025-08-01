class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=1/30):
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.dt = dt  # 时间步长

    def set_Kp(self, Kp):
        """动态设置比例增益"""
        self.Kp = Kp

    def set_Ki(self, Ki):
        """动态设置积分增益"""
        self.Ki = Ki

    def set_Kd(self, Kd):
        """动态设置微分增益"""
        self.Kd = Kd