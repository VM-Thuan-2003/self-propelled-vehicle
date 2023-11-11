import cv2
import numpy as np
import matplotlib.pyplot as plt
class lane:
    def __init__(self,input) -> None:
        self.width = input.width
        self.height = input.height
        self.left_line = np.array([])
        self.right_line = np.array([])
    def canny(self,img):
        canny = cv2.Canny(img, 50, 150)
        return canny
    def region_of_interest(self, image):
        polygons = np.array([[(0,300),(self.width,self.height),(320,110)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask,polygons,255)
        masked_image = cv2.bitwise_and(image,mask)
        return masked_image
    def houghLines(self, cropped_canny):
        return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    def addWeighted(self, frame, line_image):
        return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None: # to check if any line is detected, lines- 3-d array
            for line in lines:
                print(line)
                # x1, y1, x2, y2 = line.reshape(4)
                # # line- draws a line segment connecting 2 points, color of the line, line density
                # cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
        return image
    def make_coordinates(self, image, line_parameters):
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.1,0
        y1 = self.width
        y2 = int(y1*(3/5))
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return np.array([x1,y1,x2,y2])
    def average_slope_intercept(self, image, lines):
        left_fit = []  # Define left_fit as an empty list
        right_fit = []  # Define right_fit as an empty list

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            self.left_line = lane.make_coordinates(self, image, left_fit_average)
            # print("left",self.left_line)
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            self.right_line = lane.make_coordinates(self, image, right_fit_average)
            # print("right",self.right_line)
        if(self.right_line is not [] and self.left_line is not []):
            # print("done",self.right_line,self.left_line)
            return np.array([self.left_line,self.right_line],dtype=object)
        else:
            return np.array([[0,0,0,0],[0,0,0,0]],dtype=object)
    def detect(self,img):
        img_canny = lane.canny(self,img)
        img_region_of_interest = lane.region_of_interest(self, img_canny)
        lines = lane.houghLines(self,img_region_of_interest)
        averaged_lines = lane.average_slope_intercept(self, img, lines)
        # print(averaged_lines)
        line_image = lane.display_lines(self, img, averaged_lines)
        combo_image = lane.addWeighted(self, img, line_image)
        return combo_image
class sign:
    def __init__(self,input) -> None:
        self.width = input.width
        self.height = input.height
    def detect(self,img):
        # print(img)
        return img
class barrier:
    def __init__(self,input) -> None:
        self.width = input.width
        self.height = input.height
    def detect(self,img):
        # print(img)
        return img

class handle(lane, sign, barrier):
    def __init__(self) -> None:
        self.width = 640 # row
        self.height = 360 # col
        self.lane = lane(self)
        self.sign = sign(self)
        self.barrier = barrier(self)
    def show(self,img,name):
        img = cv2.resize(img, [self.width,self.height])
        cv2.imshow(name,img)
    def preProcess(self,img):
        alpha=1.1   # alpha: Hệ số độ tương phản (thay đổi alpha để điều chỉnh).
        beta=20      # beta: Hệ số độ sáng (thay đổi beta để điều chỉnh).
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # enhanced_image = cv2.convertScaleAbs(smooth_image, alpha=alpha, beta=beta)
        return smooth_image
    def handle(self, img):
        preProcess = handle.preProcess(self,img)
        detect_lane = self.lane.detect(preProcess)
        handle.show(self,preProcess,"preProcess")
        handle.show(self,detect_lane,"detectLane")