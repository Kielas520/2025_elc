class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=1/30):
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.dt = dt  # 时间步长
        self.prev_error = 0  # 上一次误差
        self.integral = 0  # 积分项累积
        self.max_integral = 100  # 积分项上限，防止积分饱和
        self.min_integral = -100  # 积分项下限

    def update(self, error, dt):
        """更新 PID 控制输出
        :param error: 当前误差（目标角度 - 实际角度）
        :param dt: 时间步长
        :return: 控制输出（角度调整量）
        """
        self.dt = dt
        # 比例项
        proportional = self.Kp * error
        # 积分项
        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_integral), self.min_integral)  # 限制积分项
        integral = self.Ki * self.integral
        # 微分项
        derivative = self.Kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        # 总输出
        output = proportional + integral + derivative
        return output

    def set_Kp(self, Kp):
        """动态设置比例增益"""
        self.Kp = Kp

    def set_Ki(self, Ki):
        """动态设置积分增益"""
        self.Ki = Ki

    def set_Kd(self, Kd):
        """动态设置微分增益"""
        self.Kd = Kd