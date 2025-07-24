import serial as s
import struct
import time

class Serial:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=1, write_timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.ser = None
        self.open_port()

    def open_port(self):
        """打开串口"""
        try:
            self.ser = s.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=s.EIGHTBITS,
                parity=s.PARITY_NONE,
                stopbits=s.STOPBITS_ONE,
                timeout=self.timeout,
                write_timeout=self.write_timeout
            )
            print(f"成功打开串口: {self.port}")
        except Exception as e:
            print(f"打开串口失败: {str(e)}")
            time.sleep(1)
            self.open_port()

    def send_data(self, yaw=0.0, pitch=0.0):
        """发送 float 类型的 yaw 和 pitch（各 4 字节）"""
        try:
            if not self.ser or not self.ser.is_open:
                self.reopen_port()
                return

            # 协议帧结构
            header_1 = 0xAA
            header_2 = 0x55
            command_id = 0x01
            length = 0x08  # yaw + pitch = 8 字节
            checksum = command_id ^ length

            # 将 float 转为 4 字节 bytes，并逐字节计算校验
            yaw_bytes = struct.pack('<f', yaw)
            pitch_bytes = struct.pack('<f', pitch)
            for byte in yaw_bytes + pitch_bytes:
                checksum ^= byte

            tail_1 = 0x0D
            tail_2 = 0x0A

            # 打包数据（分部分构造，避免直接拼接整数 checksum）
            packet = (
                struct.pack("<BBBB", header_1, header_2, command_id, length) +
                yaw_bytes +
                pitch_bytes +
                struct.pack("<B", checksum) +  # 确保 checksum 是 1 字节
                struct.pack("<BB", tail_1, tail_2)
            )

            self.ser.write(packet)
            print(f"发送数据: {packet.hex(' ')} | 长度: {len(packet)} bytes")
        except Exception as e:
            print(f"发送数据时出错: {str(e)}")
            self.reopen_port()

    def reopen_port(self):
        """重新打开串口"""
        print("尝试重新打开串口...")
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.open_port()
        except Exception as e:
            print(f"重新打开串口失败: {str(e)}")
            time.sleep(1)
            self.reopen_port()

    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")