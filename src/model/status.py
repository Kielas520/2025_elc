# import sys
# import signal
# import Hobot.GPIO as GPIO
# import time

# def signal_handler(signal, frame):
#     sys.exit(0)

# # 定义使用的GPIO通道为37
# input_pin = 37 # BOARD 编码 37 

# GPIO.setwarnings(False)

# def main():
#     prev_value = None

#     # 设置管脚编码模式为硬件编号 BOARD
#     GPIO.setmode(GPIO.BOARD)
#     # 设置为输入模式
#     GPIO.setup(input_pin, GPIO.IN)

#     print("Starting demo now! Press CTRL+C to exit")
#     try:
#         while True:
#             # 读取管脚电平
#             value = GPIO.input(input_pin)
#             if value != prev_value:
#                 if value == GPIO.HIGH:
#                     value_str = "HIGH"
#                 else:
#                     value_str = "LOW"
#                 print("Value read from pin {} : {}".format(input_pin, value_str))
#                 prev_value = value
#             time.sleep(1)
#     finally:
#         GPIO.cleanup()

# if __name__=='__main__':
#     signal.signal(signal.SIGINT, signal_handler)
#     main()


import sys
import signal
import Hobot.GPIO as GPIO
import time

def signal_handler(signal, frame):
    sys.exit(0)

# 定义使用的GPIO通道：
# 31号作为输出，可以点亮一个LED
# 37号作为输入，可以接一个按钮
led_pin = 16  # BOARD 编码 31
but_pin = 36 # BOARD 编码 37

# 禁用警告信息
GPIO.setwarnings(False)

def main():
    prev_value = None

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(led_pin, GPIO.OUT)  # LED pin set as output
    GPIO.setup(but_pin, GPIO.IN)  # Button pin set as input

    # Initial state for LEDs:
    GPIO.output(led_pin, GPIO.HIGH)
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            curr_value = GPIO.input(but_pin)
            if curr_value != prev_value:
                GPIO.output(led_pin, curr_value)
                prev_value = curr_value
                print("Outputting {} to Pin {}".format(curr_value, led_pin))
            
    finally:
        # GPIO.cleanup()  # cleanup all GPIO
        GPIO.output(led_pin, GPIO.LOW)
    #finally:
        

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
