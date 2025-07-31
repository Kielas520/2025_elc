import Hobot.GPIO as GPIO
class GPIN:
    def __init__(self, pin, mode):
        self.pin = pin
        self.times = 0
        self.status = GPIO.LOW
        self.prev_value = None
        self.current_value = None
        self.mode = mode
        GPIO.setmode(GPIO.BOARD)
        if self.mode == 1:
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
        elif self.mode == 0:
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setwarnings(False)

    def flash(self):
        self.times += 1
        self.times = self.times % 20
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
        if self.mode == 1:
            print("wrong_pin_type!!!!!!!!!!!!!!!!!!!!!!!1")
            return None
        
        self.current_value = GPIO.input(self.pin)
        if self.current_value != self.prev_value and self.current_value == GPIO.HIGH:  # 仅在高电平触发
            self.prev_value = self.current_value
            # 支持 0、1、2 任务切换
            task = (task + 1) % 2  # 假设任务在 0、1、2 之间循环
            return task
        self.prev_value = self.current_value
        return None
