# serial.py
import serial as s
import struct
import time

class Serial:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=1, write_timeout=1):
        self.ser = s.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=s.EIGHTBITS,  # 明确指定数据位数
            parity=s.PARITY_NONE,  # 校验位
            stopbits=s.STOPBITS_ONE,  # 停止位
            timeout=timeout,  # 读取超时
            write_timeout=write_timeout  # 写入超时
        )

    def send_data(self, yaw=0, pitch=0):
        """处理目标信息并通过串口发送"""
        try:
            header_1 = 0xAA
            header_2 = 0x55
            command_id = 0x02
            tail_1 = 0x0D
            tail_2 = 0x0A
            packet = struct.pack(
                "<BBBffBB",
                header_1,
                header_2,
                command_id,
                float(yaw),
                float(pitch),
                tail_1,
                tail_2
            )
            self.ser.write(packet)  # 修正为 self.ser
            print(f'数据:{packet} ' + f'长度: {len(packet)} bytes')
        except Exception as e:
            print(f"发送数据时出错: {str(e)}")
            self.reopen_port()
        
    def reopen_port(self):
        """重新打开串口"""
        print("尝试重新打开串口")
        try:
            if self.ser.is_open:
                self.ser.close()
            self.ser.open()
            print("成功重新打开串口")
        except Exception as e:
            print(f"重新打开串口时出错: {str(e)}")
            time.sleep(1)
            self.reopen_port()