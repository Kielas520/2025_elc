import cv2

class Camera:
    def __init__(self, index = 0, format = 'MJPG', width = 1920, height = 1080, fps = 120):
        self.cam = cv2.VideoCapture(index)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, height)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, width)
        self.cam.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ret, frame = self.cam.read()
        return ret, frame
