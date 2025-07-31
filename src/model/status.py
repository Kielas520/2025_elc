import Hobot.GPIO as GPIO
import time
class GPIN:
    def __init__(self, pin, mode):
        self.pin = pin
        self.times = 0
        self.status = GPIO.LOW
        self.prev_value = 1
        self.mode = mode
        GPIO.setmode(GPIO.BOARD)
        if self.mode == 1:
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
        elif self.mode == 0:
            GPIO.setup(self.pin, GPIO.IN)
        GPIO.setwarnings(False)

    def flash(self):
        self.times += 1
        self.times = self.times % 10
        if self.times == 0:
            if self.status == GPIO.LOW:
                self.status = GPIO.HIGH
            else:
                self.status = GPIO.LOW
        GPIO.output(self.pin, self.status)

    def set_value(self, value):
        if self.mode == 1:  # 仅输出模式支持设置值
            GPIO.output(self.pin, GPIO.HIGH if value == 1 else GPIO.LOW)
        else:
            print(f"错误: 引脚 {self.pin} 是输入模式，无法设置值")

    def button_callback(self, task):
        # 检查模式是否为1，如果是则返回错误
        if self.mode == 1:
            print("错误的引脚类型!!!!!!!!!!!!!!!!!!!!!!!")
            return None
        
        # 检测按钮按下（低电平）
        return int(GPIO.input(self.pin) == GPIO.LOW)
