import serial
import enum
import time

# 定义 SysParams_t 枚举类
class SysParams(enum.Enum):
    S_VER = 1
    S_RL = 2
    S_PID = 3
    S_VBUS = 4
    S_CPHA = 5
    S_ENCL = 6
    S_TPOS = 7
    S_VEL = 8
    S_CPOS = 9
    S_PERR = 10
    S_FLAG = 11
    S_ORG = 12
    S_Conf = 13
    S_State = 14

class MotorController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1, motor_id=1):
        """初始化电机控制器并打开串口
        :param port: 串口地址，默认为 '/dev/ttyUSB0'
        :param baudrate: 波特率，默认为 115200
        :param timeout: 超时时间，默认为 1 秒
        :param motor_id: 电机ID，默认为 1
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.motor_id = motor_id
        self.serial_port = None
        self._init_serial()

    def _init_serial(self):
        """初始化串口连接"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
        except AttributeError:
            raise Exception("串口初始化失败：pyserial 模块可能未正确安装或被覆盖")
        except serial.SerialException:
            raise Exception(f"打开串口 {self.port} 失败")

    def emm_v5_reset_curpos_to_zero(self, addr: int = None):
        """将当前位置清零
        :param addr: 电机地址，默认为实例的 motor_id
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0x0A, 0x6D, 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_reset_clog_pro(self, addr: int = None):
        """解除堵转保护
        :param addr: 电机地址，默认为实例的 motor_id
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0x0E, 0x52, 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_read_sys_params(self, addr: int = None, s: SysParams = None):
        """读取系统参数
        :param addr: 电机地址，默认为实例的 motor_id
        :param s: 系统参数类型
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = [addr]
            if s == SysParams.S_VER:
                cmd.append(0x1F)
            elif s == SysParams.S_RL:
                cmd.append(0x20)
            elif s == SysParams.S_PID:
                cmd.append(0x21)
            elif s == SysParams.S_VBUS:
                cmd.append(0x24)
            elif s == SysParams.S_CPHA:
                cmd.append(0x27)
            elif s == SysParams.S_ENCL:
                cmd.append(0x31)
            elif s == SysParams.S_TPOS:
                cmd.append(0x33)
            elif s == SysParams.S_VEL:
                cmd.append(0x35)
            elif s == SysParams.S_CPOS:
                cmd.append(0x36)
            elif s == SysParams.S_PERR:
                cmd.append(0x37)
            elif s == SysParams.S_FLAG:
                cmd.append(0x3A)
            elif s == SysParams.S_ORG:
                cmd.append(0x3B)
            elif s == SysParams.S_Conf:
                cmd.extend([0x42, 0x6C])
            elif s == SysParams.S_State:
                cmd.extend([0x43, 0x7A])
            cmd.append(0x6B)
            self.serial_port.write(bytes(cmd))
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_modify_ctrl_mode(self, addr: int = None, svF: bool = False, ctrl_mode: int = 0):
        """修改开环/闭环控制模式
        :param addr: 电机地址，默认为实例的 motor_id
        :param svF: 是否存储标志
        :param ctrl_mode: 控制模式
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0x46, 0x69, int(svF), ctrl_mode, 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_en_control(self, addr: int = None, state: bool = False, snF: bool = False):
        """使能信号控制
        :param addr: 电机地址，默认为实例的 motor_id
        :param state: 使能状态
        :param snF: 多机同步标志
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0xF3, 0xAB, int(state), int(snF), 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_vel_control(self, addr: int = None, dir: int = 0, vel: int = 0, acc: int = 0, snF: bool = False):
        """速度模式
        :param addr: 电机地址，默认为实例的 motor_id
        :param dir: 方向
        :param vel: 速度
        :param acc: 加速度
        :param snF: 多机同步标志
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([
                addr, 0xF6, dir,
                (vel >> 8) & 0xFF, vel & 0xFF,
                acc, int(snF), 0x6B
            ])
            self.serial_port.write(cmd)
            #return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_pos_control(self, addr: int = None, dir: int = 0, vel: int = 0, acc: int = 0, clk: int = 0, raF: bool = False, snF: bool = False):
        """位置模式
        :param addr: 电机地址，默认为实例的 motor_id
        :param dir: 方向
        :param vel: 速度
        :param acc: 加速度
        :param clk: 脉冲数
        :param raF: 相位/绝对标志
        :param snF: 多机同步标志
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([
                addr, 0xFD, dir,
                (vel >> 8) & 0xFF, vel & 0xFF,
                acc,
                (clk >> 24) & 0xFF, (clk >> 16) & 0xFF, (clk >> 8) & 0xFF, clk & 0xFF,
                int(raF), int(snF), 0x6B
            ])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_stop_now(self, addr: int = None, snF: bool = False):
        """立即停止
        :param addr: 电机地址，默认为实例的 motor_id
        :param snF: 多机同步标志
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0xFE, 0x98, int(snF), 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_synchronous_motion(self, addr: int = None):
        """多机同步运动
        :param addr: 电机地址，默认为实例的 motor_id
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0xFF, 0x66, 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_origin_set_o(self, addr: int = None, svF: bool = False):
        """设置单圈回零的零点位置
        :param addr: 电机地址，默认为实例的 motor_id
        :param svF: 是否存储标志
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0x93, 0x88, int(svF), 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_origin_modify_params(self, addr: int = None, svF: bool = False, o_mode: int = 0, o_dir: int = 0, o_vel: int = 0, o_tm: int = 0, sl_vel: int = 0, sl_ma: int = 0, sl_ms: int = 0, potF: bool = False):
        """修改回零参数
        :param addr: 电机地址，默认为实例的 motor_id
        :param svF: 是否存储标志
        :param o_mode: 回零模式
        :param o_dir: 回零方向
        :param o_vel: 回零速度
        :param o_tm: 回零超时时间
        :param sl_vel: 无限位碰撞回零检测转速
        :param sl_ma: 无限位碰撞回零检测电流
        :param sl_ms: 无限位碰撞回零检测时间
        :param potF: 上电自动触发回零
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([
                addr, 0x4C, 0xAE, int(svF), o_mode, o_dir,
                (o_vel >> 8) & 0xFF, o_vel & 0xFF,
                (o_tm >> 24) & 0xFF, (o_tm >> 16) & 0xFF, (o_tm >> 8) & 0xFF, o_tm & 0xFF,
                (sl_vel >> 8) & 0xFF, sl_vel & 0xFF,
                (sl_ma >> 8) & 0xFF, sl_ma & 0xFF,
                (sl_ms >> 8) & 0xFF, sl_ms & 0xFF,
                int(potF), 0x6B
            ])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_origin_trigger_return(self, addr: int = None, o_mode: int = 0, snF: bool = False):
        """触发回零
        :param addr: 电机地址，默认为实例的 motor_id
        :param o_mode: 回零模式
        :param snF: 多机同步标志
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0x9A, o_mode, int(snF), 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_origin_interrupt(self, addr: int = None):
        """强制中断并退出回零
        :param addr: 电机地址，默认为实例的 motor_id
        """
        addr = self.motor_id if addr is None else addr
        try:
            cmd = bytes([addr, 0x9C, 0x48, 0x6B])
            self.serial_port.write(cmd)
            return self.serial_port.read(10)
        except serial.SerialException:
            raise

    def emm_v5_move_to_angle(self, addr: int = None, angle_deg: float = 0.0, vel_rpm: int = 0, acc: int = 0, abs_mode: bool = False):
        """
        按角度移动电机
        :param addr: 电机地址，默认为实例的 motor_id
        :param angle_deg: 目标角度（度）
        :param vel_rpm: 速度（RPM）
        :param acc: 加速度（0-255）
        :param abs_mode: 是否为绝对模式（True 绝对，False 相对）
        :return: 串口响应
        """
        addr = self.motor_id if addr is None else addr
        try:
            dir = 0  # 默认顺时针 (CW)
            if angle_deg > 0:
                dir = 0  # 顺时针
            else:
                dir = 1  # 逆时针 (CCW)
                angle_deg = -angle_deg  # 取绝对值

            # 计算脉冲数 (假设每 360 度 51200 脉冲，根据协议调整)
            clk = int(angle_deg * 51200.0 / 360.0)

            # 调用位置控制函数
            return self.emm_v5_pos_control(addr, dir, vel_rpm, acc, clk, abs_mode, False)
        except serial.SerialException:
            raise

    def close(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

def test_motor_control(motor: MotorController, addr: int = None):
    """
    测试电机控制功能
    :param motor: MotorController 实例
    :param addr: 电机地址，默认为实例的 motor_id
    """
    addr = motor.motor_id if addr is None else addr
    try:
        # 测试1：读取系统版本
        motor.emm_v5_read_sys_params(addr, SysParams.S_VER)

        # 测试2：重置当前位置
        motor.emm_v5_reset_curpos_to_zero(addr)

        # 测试3：设置闭环控制模式（存储）
        motor.emm_v5_modify_ctrl_mode(addr, svF = True, ctrl_mode = 2)

        # 测试4：使能电机
        motor.emm_v5_en_control(addr, state = True, snF = False)

        # 测试5：按角度移动（90°，顺时针）
        motor.emm_v5_move_to_angle(addr, angle_deg = 90.0, vel_rpm = 1000, acc = 50, abs_mode = False)
        time.sleep(2)  # 运行2秒

        # 测试6：立即停止
        motor.emm_v5_stop_now(addr, snF = False)

        # 测试7：位置模式（相对运动，10000脉冲）
        motor.emm_v5_pos_control(addr, dir = 0, vel = 1000, acc = 50, clk = 10000, raF = False, snF = False)

        # 测试8：触发回零（单圈就近）
        motor.emm_v5_origin_trigger_return(addr, o_mode = 0, snF = False)

    except Exception:
        pass
    finally:
        motor.close()

if __name__ == "__main__":
    # 创建电机控制器实例，指定串口和电机ID
    motor = MotorController(port = '/dev/ttyS1', motor_id = 1)
    try:
        test_motor_control(motor)
    except Exception:
        motor.close()