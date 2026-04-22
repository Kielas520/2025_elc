import serial
import enum
import time
import struct
import sys

# 定义系统参数枚举，对应树莓派例程中的 key
'''
枚举是为了让代码可读性更强
S_VER 对应 0x1F 是因为协议规定读取版本号的命令字是 0x1F
'''
class SysParams(enum.Enum):
    S_VER = 'S_VER'   # 0x1F
    S_RL = 'S_RL'     # 0x20
    S_PID = 'S_PID'   # 0x21
    S_VBUS = 'S_VBUS' # 0x24
    S_CPHA = 'S_CPHA' # 0x27
    S_ENCL = 'S_ENCL' # 0x31
    S_TPOS = 'S_TPOS' # 0x33
    S_VEL = 'S_VEL'   # 0x35
    S_CPOS = 'S_CPOS' # 0x36
    S_PERR = 'S_PERR' # 0x37
    S_FLAG = 'S_FLAG' # 0x3A
    S_ORG = 'S_ORG'   # 0x3B
    S_Conf = 'S_Conf' # 0x42 + 0x6C
    S_State = 'S_State' # 0x43 + 0x7A

class EmmMotor:
    '''参数:
    port : 串口端口号
    baudrate : 波特率
    timeout : 读取超时时间(秒)
    motor_id : 电机地址ID
    '''
    def __init__(self, port='COM', baudrate=115200, timeout=1, motor_id=1):
        # 保留参数
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.motor_id = motor_id
        # 初始化
        self.serial_port = None
        # 打开串口
        self._init_serial()

    def _init_serial(self):
        '''打开串口的操作'''
        # 用try有利于检查错误
        try:
            '''创建串口对象'''
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,      # 八位数据
                parity=serial.PARITY_NONE,       # 无校验位
                stopbits=serial.STOPBITS_ONE     # 1位停止位
            )
            # 清空缓冲区，防止残留数据干扰
            if self.serial_port.is_open:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()

        except AttributeError:
            raise Exception("串口初始化失败：pyserial 模块可能未正确安装或被覆盖")
        except serial.SerialException as e:
            raise Exception(f"打开串口 {self.port} 失败: {e}")
    
    # 把字节流扔进串口
    def _send_cmd(self, cmd_bytes):
        # 检查串口有没有打开
        if not self.serial_port or not self.serial_port.is_open:
            raise Exception("串口未打开")
        
        # 发送数据
        self.serial_port.write(cmd_bytes)

    '''实现核心通信协议''' 
    def emm_v5_read_sys_params(self, addr=None, s: SysParams = None):
        """
        读取系统参数
        参考树莓派例程 Emm_V5_Read_Sys_Params
        """
        addr = self.motor_id if addr is None else addr
        cmd = bytearray()
        cmd.append(addr)
        
        # 功能码映射
        func_codes = {
            SysParams.S_VER: 0x1F,
            SysParams.S_RL: 0x20,
            SysParams.S_PID: 0x21,
            SysParams.S_VBUS: 0x24,
            SysParams.S_CPHA: 0x27,
            SysParams.S_ENCL: 0x31,
            SysParams.S_TPOS: 0x33,
            SysParams.S_VEL: 0x35,
            SysParams.S_CPOS: 0x36,
            SysParams.S_PERR: 0x37,
            SysParams.S_FLAG: 0x3A,
            SysParams.S_ORG: 0x3B,
            SysParams.S_Conf: 0x42,
            SysParams.S_State: 0x43
        }
        
        if s not in func_codes:
            raise ValueError(f"未知的参数类型: {s}")
            
        cmd.append(func_codes[s])
        
        # 特殊参数需要辅助码
        if s == SysParams.S_Conf:
            cmd.append(0x6C)
        elif s == SysParams.S_State:
            cmd.append(0x7A)
         
        cmd.append(0x6B) # 校验字节
        
        self._send_cmd(bytes(cmd))
        
        # 读取响应 (根据参数不同，返回长度可能不同，这里尝试读取一定长度)
        # 注意：实际项目中需要根据具体协议文档确定返回字节数
        time.sleep(0.01) # 等待驱动器处理
        return self.serial_port.read(32)

    def emm_v5_reset_curpos_to_zero(self, addr=None):
        '''位置清零'''
        addr = self.motor_id if addr is None else addr
        cmd = bytes([
            addr,
            0x0A, 
            0x6D,
            0x6B
        ])
        self._send_cmd(cmd)

    def emm_v5_reset_clog_pro(self, addr=None):
        '''解除堵转保护'''
        addr = self.motor_id if addr is None else addr

        # 组装命令
        cmd = bytes([
            addr,   # 地址
            0x0E,   # 命令字
            0x52,   # 数据位
            0x6B    # 校验/结束符
        ])
        self._send_cmd(cmd)

    def emm_v5_stop_now(self, addr=None, snF=False):
        '''立刻停止'''
        addr = self.motor_id if addr is None else addr

        cmd = bytes([
            addr,
            0xFE,
            0x01 if snF else 0x00, 
            0x6B
        ])
        self._send_cmd(cmd)

    def emm_v5_en_control(self, addr=None, state=False, snF=False):
        '''
        使能信号控制
        param state: True为使能(通电有力), False为去使能(断电无力)
        param snF: True为同步触发模式, False为立即执行
        '''
        addr = self.motor_id if addr is None else addr

        cmd = bytes([
            addr,
            0xF3,
            0xAB,
            0x01 if state else 0x00,  # 使能位：1=开, 0=关
            0x01 if snF else 0x00,    # 同步位：1=同步, 0=立即
            0x6B
        ])
        self._send_cmd(cmd)

    def emm_v5_modify_ctrl_mode(self, addr=None, svF=False, ctrl_mode=0):
        '''
        修改开环/闭环控制模式
        param svF: True表示将设置保存到Flash(断电不丢失), False仅本次生效
        '''
        addr = self.motor_id if addr is None else addr
        cmd = bytes([
            addr,
            0x46,
            0x01 if svF else 0x00,
            ctrl_mode,
            0x6B
        ])
        self._send_cmd(cmd)

    def emm_v5_vel_control(self, addr=None, dir=0, vel=0, acc=0, snF=False):
        '''
        速度模式控制
        param dir: 方向 (0=正转, 1=反转)
        param vel: 速度值 (0-65535)，单位通常是 RPM 或内部单位
        param acc: 加速度 (0-255)
        param snF: 同步标志
        '''
        # 速度 vel 是 16位整数，需要拆成两个 8位字节
        vel_high = (vel >> 8) & 0xFF  # 高8位
        vel_low = vel & 0xFF          # 低8位
        addr = self.motor_id if addr is None else addr
        cmd = bytes([
            addr,
            0xF6, 
            dir & 0xFF,
            vel_high,
            vel_low,
            acc & 0xFF,
            0x01 if snF else 0x00, 
            0x6B
        ])
        self._send_cmd(cmd)

    def emm_v5_pos_control(self, addr=None, snF=False, raF=False, dir=0, vel=0, acc=0, clk=0):         
        '''
        位置控制模式
        协议格式
        [地址, 0xFD, dir, vel_h, vel_l, acc, clk_3, clk_2, clk_1, clk_0, raF, snF, 0x6B]
        param clk: 目标脉冲数/位置值 (32位整数)
        param raF: True=绝对位置, False=相对位置
        '''
        vel_high = (vel >> 8) & 0xFF  # 高8位
        vel_low = vel & 0xFF          # 低8位
        # 位置（脉冲数）是一个 32位整数 (int32)，所以需要拆成 4个字节
        clk_3 = (clk >> 24) & 0xFF  # 最高8位
        clk_2 = (clk >> 16) & 0xFF
        clk_1 = (clk >> 8) & 0xFF
        clk_0 = clk & 0xFF          # 最低8位
        addr = self.motor_id if addr is None else addr
        cmd = bytes([
            addr,
            0xFD,
            dir & 0xFF,
            vel_high, 
            vel_low, 
            acc, 
            clk_3, 
            clk_2, 
            clk_1, 
            clk_0, 
            0x01 if raF else 0x00,  # raF
            0x01 if snF else 0x00,  # snF
            0x6B
        ])
        self._send_cmd(cmd)

    def emm_v5_origin_trigger_return(self, addr=None, o_mode=0, snF=False):
        '''
        回零模式
        协议格式
        [地址, 0x9A, o_mode(回零模式), snF(同步), 0x6B]
        '''
        addr = self.motor_id if addr is None else addr
        cmd = bytes([
            addr,
            0x9A,       # 修正：原代码漏掉了命令字 0x9A，根据注释补充
            o_mode & 0xFF,
            0x01 if snF else 0x00,
            0x6B
        ])
        self._send_cmd(cmd)

    # def emm_v5_synchronous_motion(self):

    def get_current_position_angle(self, addr=None):
        """
        读取当前电机位置并转换为角度
        返回: 角度 (float)
        """
        addr = self.motor_id if addr is None else addr
        # 获取原始字节数据, S_CPOS 对应的是当前脉冲位置
        data = self.emm_v5_read_sys_params(addr, SysParams.S_CPOS)

        # 数据校验与解析, 协议规定返回至少要有 7 个字节：[ Addr, Func, Dir, Pos_3, Pos_2, Pos_1, Pos_0]
        if not data or len(data) < 7:
            return None
        
        # 检查返回的地址和功能码是否正确，防止读到垃圾数据
        if data[0] != addr or data[1] != 0x36:        # 0x36 是 S_CPOS 的功能码
            return None
        
        # 获取旋转方向
        direction_byte = data[2]

        # 获取脉冲数
        pos_int = struct.unpack('>I', data[3:7])[0] # data[3:7] 取出第4到第7个字节，'>I' 表示大端序无符号整数

        # 脉冲数转角度
        resolution = 65536.0
        angle = float(pos_int) * 360.0 / resolution

        # 处理方向
        if direction_byte != 0:
            angle = -angle    # 如果方向位不为0，说明是反转，角度取负

        return angle

    def emm_v5_move_to_angle(self, addr=None, angle_deg=0.0, vel_rpm=0, acc=0, abs_mode=False):
        """
        按角度移动电机
        :param angle_deg: 目标角度 (度)，支持负数
        :param vel_rpm: 运动速度
        :param acc: 加速度
        :param abs_mode: True=绝对位置(转到指定刻度), False=相对位置(再转多少度)
        """
        # 1. 确定地址
        addr = self.motor_id if addr is None else addr
        
        # 2. 处理方向
        # 协议中 dir=0 为正，dir=1 为反。我们根据角度的正负来决定。
        dir = 0
        if angle_deg < 0:
            dir = 1
            angle_deg = -angle_deg  # 取绝对值进行计算
            
        # 3. 数学转换：角度 -> 脉冲数 (clk)
        resolution = 65536.0
        clk = int(angle_deg * resolution / 360.0)
        
        # 4. 调用底层位置控制函数
        # raF 对应 abs_mode: True 为绝对模式，False 为相对模式
        self.emm_v5_pos_control(
            addr=addr, 
            dir=dir, 
            vel=vel_rpm, 
            acc=acc, 
            clk=clk, 
            raF=abs_mode, 
            snF=False
        )

    def close(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 请根据实际情况修改端口号
    # Windows: 'COM3', 'COM4' 等
    # Linux/Mac: '/dev/ttyUSB0', '/dev/ttyAMA0' 等
    SERIAL_PORT = 'COM20' 
    BAUDRATE = 115200
    MOTOR_ID = 1
    # ===========================================

    print(f"正在尝试连接串口: {SERIAL_PORT} ...")

    motor = None
    # 1. 初始化电机对象
    motor = EmmMotor(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=1,
        motor_id=MOTOR_ID
    )
    print("串口连接成功！")

    # 2. 读取版本号，验证通信是否正常
    print("\n--- 测试1: 读取系统版本 ---")
    ver_data = motor.emm_v5_read_sys_params(s=SysParams.S_VER)
    if ver_data and len(ver_data) >= 4:
        # 假设版本信息在返回数据的特定位置，具体需参考协议文档
        # 这里仅打印原始十六进制数据以便调试
        print(f"版本响应数据 (Hex): {ver_data.hex()}")
    else:
        print("读取版本失败或无响应，请检查接线和ID。")
        sys.exit(1)

    # 3. 读取当前角度
    print("\n--- 测试2: 读取当前角度 ---")
    current_angle = motor.get_current_position_angle()
    if current_angle is not None:
        print(f"当前角度: {current_angle:.2f} 度")
    else:
        print("读取角度失败")

    # 1. 使能电机
    motor.emm_v5_en_control(state=True)

    # 2. 绝对模式：直接转到 90 度
    print("转到 90 度...")
    motor.emm_v5_move_to_angle(angle_deg=90.0, vel_rpm=500, acc=100, abs_mode=True)
    time.sleep(2)

    # 3. 读取验证
    pos = motor.get_current_position_angle()
    print(f"当前角度: {pos:.2f}")

    # 4. 相对模式：再转 -45 度（往回转 45 度）
    print("相对转动 -45 度...")
    motor.emm_v5_move_to_angle(angle_deg=-45.0, vel_rpm=500, acc=100, abs_mode=False)
    time.sleep(2)

    # 5. 停止并去使能
    motor.emm_v5_stop_now()
    motor.emm_v5_en_control(state=False)