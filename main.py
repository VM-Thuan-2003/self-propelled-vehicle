import cv2
import numpy as np

class lane:
    def __init__(self) -> None:
        pass
    def detect(self,img):
        # print(img)
        return img
class sign:
    def __init__(self) -> None:
        pass
    def detect(self,img):
        # print(img)
        return img
class barrier:
    def __init__(self) -> None:
        pass
    def detect(self,img):
        # print(img)
        return img

class handle(lane, sign, barrier):
    def __init__(self) -> None:
        self.width = 728 # row
        self.height = 512 # col
        self.lane = lane
        self.sign = sign
        self.barrier = barrier
    def show(self,img,name):
        img = cv2.resize(img, [self.width,self.height])
        cv2.imshow(name,img)
    def preProcess(self,img):
        alpha=1.1   # alpha: Hệ số độ tương phản (thay đổi alpha để điều chỉnh).
        beta=20      # beta: Hệ số độ sáng (thay đổi beta để điều chỉnh).
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        # enhanced_image = cv2.convertScaleAbs(smooth_image, alpha=alpha, beta=beta)
        return smooth_image
    def handle(self, img):
        preProcess = handle.preProcess(self,img)
        detect_lane = self.lane.detect(self,preProcess)
        handle.show(self,preProcess,"preProcess")
        handle.show(self,detect_lane,"detectLane")