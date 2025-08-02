import time
import serial
import struct
from typing import Tuple


class EmmMotor:
    """
    面向对象封装：Emm V5 电机驱动控制类
    支持通过串口发送指令、读取状态、位置控制、速度控制等。
    """

    # 功能码映射表
    FUNC_CODES = {
        "S_VER": 0x1F,
        "S_RL": 0x20,
        "S_PID": 0x21,
        "S_VBUS": 0x24,
        "S_CPHA": 0x27,
        "S_ENCL": 0x31,
        "S_TPOS": 0x33,
        "S_VEL": 0x35,
        "S_CPOS": 0x36,  # 当前位置
        "S_PERR": 0x37,
        "S_FLAG": 0x3A,
        "S_ORG": 0x3B,
        "S_Conf": 0x42,
        "S_State": 0x43,
    }

    def __init__(self, addr: int, serial_port: serial.Serial):
        """
        初始化电机对象
        :param addr: 电机地址 (1, 2, ...)
        :param serial_port: 已打开的 serial.Serial 实例
        """
        if addr < 1:
            raise ValueError("电机地址必须 >= 1")
        self.addr = addr
        self.ser = serial_port

    def _send(self, cmd: bytes):
        """内部方法：发送命令到串口"""
        self.ser.write(cmd)

    def _build_cmd(self, *args) -> bytearray:
        """构建带地址、数据和校验的命令帧"""
        cmd = bytearray([self.addr, *args, 0x6B])
        return cmd

    def read_sys_param(self, param: str):
        """读取系统参数"""
        if param not in self.FUNC_CODES:
            raise ValueError(f"无效的参数类型: {param}")
        cmd = self._build_cmd(self.FUNC_CODES[param])
        self._send(cmd)

    def reset_current_position_to_zero(self):
        """将当前位置清零"""
        cmd = self._build_cmd(0x0A, 0x6D)
        self._send(cmd)

    def reset_clog_protection(self):
        """解除堵转保护"""
        cmd = self._build_cmd(0x0E, 0x52)
        self._send(cmd)

    def modify_control_mode(self, svF: bool, ctrl_mode: int):
        """修改控制模式"""
        cmd = self._build_cmd(0x46, 0x69, 0x01 if svF else 0x00, ctrl_mode)
        self._send(cmd)

    def enable_control(self, state: bool, snF: bool = False):
        """使能电机控制"""
        cmd = self._build_cmd(
            0xF3, 0xAB, 0x01 if state else 0x00, 0x01 if snF else 0x00
        )
        self._send(cmd)

    def velocity_control(
        self, direction: int, velocity: int, acceleration: int, snF: bool
    ):
        """速度控制：direction=0正转, 1反转；velocity单位RPM；acceleration"""
        cmd = self._build_cmd(
            0xF6,
            direction & 0xFF,
            (velocity >> 8) & 0xFF,
            velocity & 0xFF,
            acceleration & 0xFF,
            0x01 if snF else 0x00,
        )
        self._send(cmd)

    def position_control(
        self,
        direction: int,
        velocity: int,
        acceleration: int,
        pulses: int,
        raF: bool = False,
        snF: bool = False,
    ):
        """位置控制：pulses为脉冲数（圈数×1000）"""
        cmd = self._build_cmd(
            0xFD,
            direction & 0xFF,
            (velocity >> 8) & 0xFF,
            velocity & 0xFF,
            acceleration & 0xFF,
            (pulses >> 24) & 0xFF,
            (pulses >> 16) & 0xFF,
            (pulses >> 8) & 0xFF,
            pulses & 0xFF,
            0x01 if raF else 0x00,
            0x01 if snF else 0x00,
        )
        self._send(cmd)

    def stop_now(self, snF: bool):
        """立即停止"""
        cmd = self._build_cmd(0xFE, 0x98, 0x01 if snF else 0x00)
        self._send(cmd)

    def synchronous_motion(self):
        """执行多机同步运动（需配合其他指令）"""
        cmd = self._build_cmd(0xFF, 0x66)
        self._send(cmd)

    def origin_set_single圈_zero(self, svF: bool):
        """设置单圈回零零点位置"""
        cmd = self._build_cmd(0x93, 0x88, 0x01 if svF else 0x00)
        self._send(cmd)

    def origin_modify_params(
        self,
        svF: bool,
        o_mode: int,
        o_dir: int,
        o_vel: int,
        o_tm: int,
        sl_vel: int,
        sl_ma: int,
        sl_ms: int,
        potF: bool,
    ):
        """修改回零参数"""
        cmd = self._build_cmd(
            0x4C,
            0xAE,
            0x01 if svF else 0x00,
            o_mode & 0xFF,
            o_dir & 0xFF,
            (o_vel >> 8) & 0xFF,
            o_vel & 0xFF,
            (o_tm >> 24) & 0xFF,
            (o_tm >> 16) & 0xFF,
            (o_tm >> 8) & 0xFF,
            o_tm & 0xFF,
            (sl_vel >> 8) & 0xFF,
            sl_vel & 0xFF,
            (sl_ma >> 8) & 0xFF,
            sl_ma & 0xFF,
            (sl_ms >> 8) & 0xFF,
            sl_ms & 0xFF,
            0x01 if potF else 0x00,
        )
        self._send(cmd)

    def origin_trigger_return(self, o_mode: int, snF: bool):
        """触发回零"""
        cmd = self._build_cmd(0x9A, o_mode & 0xFF, 0x01 if snF else 0x00)
        self._send(cmd)

    def origin_interrupt(self):
        """强制中断回零"""
        cmd = self._build_cmd(0x9C, 0x48)
        self._send(cmd)

    def receive_data(self, timeout: float = 0.1) -> Tuple[str, int]:
        """接收串口返回数据"""
        start_time = time.time()
        buffer = bytearray()

        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                buffer.extend(chunk)
                start_time = time.time()  # 重置超时计时器
            time.sleep(0.001)

        hex_data = " ".join(f"{b:02x}" for b in buffer)
        return hex_data, len(buffer)

    def get_real_position(self) -> float:
        """
        获取当前电机位置角度（°）
        返回值：角度（浮点数），失败时返回 0.0
        """
        self.read_sys_param("S_CPOS")
        time.sleep(0.001)
        hex_data, _ = self.receive_data(timeout=0.1)
        return self._parse_position(hex_data)

    def _parse_position(self, hex_data: str) -> float:
        """解析 S_CPOS 返回的位置数据"""
        if not hex_data:
            return 0.0
        parts = hex_data.split()
        if len(parts) < 7:
            return 0.0

        try:
            addr = int(parts[0], 16)
            cmd = int(parts[1], 16)
            if addr != self.addr or cmd != 0x36:  # 验证地址和指令
                return 0.0

            # 提取4字节位置（大端序）
            pos_bytes = bytes(
                [
                    int(parts[3], 16),
                    int(parts[4], 16),
                    int(parts[5], 16),
                    int(parts[6], 16),
                ]
            )
            pos_value = struct.unpack(">I", pos_bytes)[0]

            # 转换为角度：360° / 65536 单位
            angle = (pos_value * 360.0) / 65536.0

            # 检查方向标志
            dir_flag = int(parts[2], 16)
            if dir_flag != 0:
                angle = -angle

            return angle
        except (IndexError, ValueError, struct.error):
            return 0.0


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 配置串口
    BAUD_RATE = 115200
    SERIAL_PORT1 = "COM13"
    SERIAL_PORT2 = "/dev/ttyS6"

    try:
        # 打开串口
        # ser1 = serial.Serial(SERIAL_PORT1, BAUD_RATE, timeout=0.1)
        ser2 = serial.Serial(SERIAL_PORT2, BAUD_RATE, timeout=0.1)
        print(f"已连接串口: {SERIAL_PORT1}, {SERIAL_PORT2}")

        # 创建电机对象
        motor1 = EmmMotor(addr=1, serial_port=ser2)  # 共用一个串口
        motor2 = EmmMotor(addr=2, serial_port=ser2)

        # 使能电机
        motor1.enable_control(state=True)
        motor2.enable_control(state=True)
        time.sleep(0.1)

        # 发送位置控制指令（电机2：正转，4000RPM，100脉冲）        
        
        motor2.position_control(
            direction=0, velocity=10, acceleration=0, pulses=0, raF=True, snF=False
        )

        time.sleep(0.01)

        motor1.position_control(
            direction=0, velocity=10, acceleration=0, pulses=0, raF=True, snF=False
        )
        time.sleep(0.01)
        time.sleep(1)


        # motor2.position_control(
        #     direction=0, velocity=10, acceleration=0, pulses=500, raF=False, snF=False
        # )

        # time.sleep(0.005)

        # motor1.position_control(
        #     direction=0, velocity=10, acceleration=0, pulses=500, raF=False, snF=False
        # )
        # time.sleep(0.01)
        # time.sleep(1)


        # motor2.position_control(
        #     direction=1, velocity=10, acceleration=0, pulses=500, raF=False, snF=False
        # )

        # time.sleep(0.005)

        # motor1.position_control(
        #     direction=1, velocity=10, acceleration=0, pulses=500, raF=False, snF=False
        # )
        # time.sleep(1)

        # 失能电机
        motor1.enable_control(state=False)
        motor2.enable_control(state=False)

        # 持续读取两个电机的位置
        print("实时位置监控 (Ctrl+C 退出):")
        while True:
            pos1 = motor1.get_real_position()
            pos2 = motor2.get_real_position()
            print(f"Motor1: {pos1:6.1f}°, Motor2: {pos2:6.1f}°")
            # print(f" Motor2: {pos2:6.1f}°")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n用户中断，停止电机...")
        motor1.stop_now(snF=False)
        motor2.stop_now(snF=False)
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        try:
            # ser1.close()
            pass
        except:
            pass
        try:
            ser2.close()
        except:
            pass
        print("串口已关闭")