import cv2
import numpy as np
import matplotlib.pyplot as plt
class lane:
    def __init__(self,input) -> None:
        self.width = input.width
        self.height = input.height
    def canny(self,img):
        # canny = cv2.Canny(img, 50, 150)
        canny = cv2.Canny(img, 70, 135) 
        return canny
    def region_selection(self, image):
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        rows, cols = image.shape[:2]
        bottom_left  = [0, rows * 0.95]
        top_left     = [300,100]
        bottom_right = [cols, rows]
        top_right    = [300,100]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    def hough_transform(self, image):
        rho = 1
        theta = np.pi/180
        threshold = 20
        minLineLength = 20
        maxLineGap = 500
        return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                            minLineLength = minLineLength, maxLineGap = maxLineGap)
    def average_slope_intercept(self, lines):
        left_lines    = [] #(slope, intercept)
        left_weights  = [] #(length,)
        right_lines   = [] #(slope, intercept)
        right_weights = [] #(length,)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane

    def pixel_points(self, y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
    
    def lane_lines(self, image, lines):
        left_lane, right_lane = lane.average_slope_intercept(self, lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line  = lane.pixel_points(self, y1, y2, left_lane)
        right_line = lane.pixel_points(self, y1, y2, right_lane)
        return left_line, right_line
    
        
    def draw_lane_lines(self, image, lines, color=[255, 0, 0], thickness=12):
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line,  color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
    def detect(self,img):
        img_canny = lane.canny(self,img)
        region = lane.region_selection(self, img_canny)
        hough = lane.hough_transform(self, region)
        result = lane.draw_lane_lines(self, img, lane.lane_lines(self, img, hough))
        return result
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
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        smooth_image = cv2.GaussianBlur(src=gray_image, ksize=(3, 5), sigmaX=0.5)
        return smooth_image
    def handle(self, img):
        preProcess = handle.preProcess(self,img)
        detect_lane = self.lane.detect(preProcess)
        handle.show(self,preProcess,"preProcess")
        handle.show(self,detect_lane,"detectLane")