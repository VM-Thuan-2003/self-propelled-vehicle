#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Import socket module
from code import interact
import math
import time
import socket
import cv2
import numpy as np
import pandas as pd

global sendBack_angle, sendBack_Speed, current_speed, current_angle

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321 # connect to the server on local computer 54321
s.connect(('127.0.0.1', PORT))

sendBack_angle = 0
sendBack_Speed = 20

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed

#-----------------PID Controller-------------------#
error_arr = np.zeros(5)
pre_t = time.time()
MAX_SPEED = 150

def PID(error, p, i, d):
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)

class Lane:
    def __init__(self,img) -> None:
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.img = img
    def scaleImg(self,img):
        scaleValue = 195  #Scale value x = 200
        return img[scaleValue:,:]
    def preImg(self,img):
        lower_green = np.array([95,0,0]).astype("uint8")
        upper_green = np.array([255,255,255]).astype("uint8")
        mask = cv2.inRange(img,lower_green,upper_green)
        lane = cv2.bitwise_and(img, img, mask=mask)
        color_select = np.copy(lane)
        red_threshold = 110
        green_threshold = 127
        blue_threshold = 110
        rgb_threshold = [red_threshold, green_threshold, blue_threshold]
        thresholds = (img[:,:,0] < rgb_threshold[0]) \
                    | (img[:,:,1] < rgb_threshold[1]) \
                    | (img[:,:,2] < rgb_threshold[2])
        color_select[thresholds] = [0,0,0]
        return color_select
    def grayImg(self, img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    def  GaussianBlur(self,img):
        kernel_size = 5
        GaussianBlur = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
        return GaussianBlur
    def binaryImg(self, img):
        threshold = 110
        _, binary_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary_thresh
    def regionMasking(self, img):
        addYSize = 0
        rectangle_coordinates = [(0, int(self.height/8) + addYSize), (self.width, self.height)]
        mask = np.zeros_like(img)*255
        cv2.rectangle(mask, rectangle_coordinates[0], rectangle_coordinates[1], (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(img, mask)
        return result
    def  cannyEdgeDetection(self,img):
        low_threshold = 150
        high_threshold = 255
        GaussianBlur = Lane.GaussianBlur(self, img)
        cannyEdgeDetection = cv2.Canny(GaussianBlur, low_threshold, high_threshold)
        return cannyEdgeDetection
    def grayCTbinImg(self,img):
        lower_green = np.array([95,0,0]).astype("uint8")
        upper_green = np.array([255,255,255]).astype("uint8")

        mask = cv2.inRange(img,lower_green,upper_green)
        lane = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(lane,cv2.COLOR_BGR2GRAY)
    def preNoneCenter(self,img,imgOrigin):
        # dis = 100
        # addXSize_1 = -dis
        # addXSize_2 = -addXSize_1
        # rectangle_coordinates = [(int(self.width/2 + addXSize_1), 0),(int(self.width/2 + addXSize_2),self.height)]
        # mask = np.zeros_like(img)*255
        # cv2.rectangle(mask, (0,0), (self.width, self.height), (255), thickness=cv2.FILLED)
        # cv2.rectangle(mask, rectangle_coordinates[0], rectangle_coordinates[1], (0,0,0), thickness=cv2.FILLED)
        # cv2.imshow("iiiiii_img",img)
        # result = cv2.bitwise_and(img, mask)
        
        # mask = np.zeros_like(img)*255
        # triangle_points = np.array([[int(self.width/2), 10], [0, self.height], [self.width, self.height]], np.int32)
        # triangle_points = triangle_points.reshape((-1, 1, 2))
        # cv2.rectangle(mask, (0,0), (self.width, self.height), (255), thickness=cv2.FILLED)
        # cv2.polylines(mask, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=2)
        # cv2.fillPoly(mask, [triangle_points], color=(0, 0, 0))
        # cv2.imshow("iiiiii_mask",mask)
        # result = cv2.bitwise_and(img, mask)
        cv2.imshow("img",img)
        cv2.imshow("imgOrigin",imgOrigin)
        
        contours_1, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour_1 = max(contours_1, key=cv2.contourArea)
        mask_1 = np.zeros_like(img)
        cv2.drawContours(mask_1, [largest_contour_1], 0, (255), thickness=cv2.FILLED)
        result_1 = np.ones_like(img) * 255
        result_1 = cv2.bitwise_and(result_1, result_1, mask=mask_1)
        _, binary_thresh_1 = cv2.threshold(result_1, 220, 255, cv2.THRESH_BINARY_INV)
        result_1 = cv2.bitwise_and(img, img, mask=binary_thresh_1)
        cv2.imshow("result_1",result_1)
        
        contours, _ = cv2.findContours(imgOrigin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(imgOrigin)
        cv2.drawContours(mask, [largest_contour], 0, (255), thickness=cv2.FILLED)
        result = np.ones_like(imgOrigin) * 255
        result = cv2.bitwise_and(result, result, mask=mask)
        cv2.imshow("result",result)
        
        #  old id 220
        _, binary_thresh = cv2.threshold(result, 220, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("mask111",binary_thresh)
        
        contours_2, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour_2 = max(contours_2, key=cv2.contourArea)
        mask_2 = np.zeros_like(binary_thresh)
        cv2.drawContours(mask_2, [largest_contour_2], 0, (255), thickness=cv2.FILLED)
        result_2 = np.ones_like(binary_thresh) * 255
        result_2 = cv2.bitwise_and(result_2, result_2, mask=mask_2)
        _, binary_thresh_2 = cv2.threshold(result_2, 220, 255, cv2.THRESH_BINARY_INV)
        result_2 = cv2.bitwise_and(binary_thresh, binary_thresh, mask=binary_thresh_2)
        _, binary_thresh_3 = cv2.threshold(result_2, 220, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("result_2",binary_thresh_3)
        
        return result
        # return img
    def contourImg(self,img,imgOrigin):
        # img1 = Lane.preNoneCenter(self,img,imgOrigin)
        # cv2.imshow("iiiiii",img1)
        cv2.imshow("qqqqqq",img)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        area=[]
        for i, contour in enumerate(contours):
            area.append(cv2.contourArea(contour))
        max_area = np.max(area)
        idx = area.index(max_area)
        # print(idx)
        cnt = contours[idx]
        # print(cnt)
        cv2.drawContours(img,[cnt],0,(0,0,0),cv2.FILLED)
        _, binary_thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        return binary_thresh
    def houghTransform(self, cannyEdgeDetection):
        _img_bgr = np.copy(Lane.scaleImg(self, self.img))
        rho = 5              # distance resolution in pixels of the Hough grid
        theta = np.pi/180    # angular resolution in radians of the Hough grid
        threshold = 50       # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 40    # maximum gap in pixels between connectable line segments
        lineP = cv2.HoughLinesP(cannyEdgeDetection, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
        if lineP is not None:
            for i in range(0, len(lineP)):
                l = lineP[i][0]
                cv2.line(_img_bgr,(l[0],l[1]),(l[2],l[3]),(255,255,255),8,cv2.LINE_AA)
        _, binary_thresh = cv2.threshold(_img_bgr, 220, 255, cv2.THRESH_BINARY)
        binary_thresh = Lane.grayImg(self,binary_thresh)
        # binary_thresh = Lane.preNoneCenter(self,binary_thresh)
        return binary_thresh
    def combineImg(self, img1, img2):
        return cv2.bitwise_or(img1, img2)
    def controlDevice(self, img):
        
        # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = max(contours, key=cv2.contourArea)
        # mask = np.zeros_like(img)
        # cv2.drawContours(mask, [largest_contour], 0, (255), thickness=cv2.FILLED)
        # result = np.ones_like(img) * 255
        # result = cv2.bitwise_and(result, result, mask=mask)
        # cv2.imshow("mask",result)

        kc_set = 68
        lineRow = img[kc_set,:]
        arr=[]
        for x,y in enumerate(lineRow):
            if y == 255:
                arr.append(x)
        arrMax = max(arr)
        arrMin = min(arr)
        center = int((arrMax + arrMin)/2)
        angle1 = math.degrees(math.atan((center - img.shape[1]/2)/(img.shape[0]-kc_set)))
        cv2.circle(img, (arrMin,kc_set), 5, (0,0,255),5)
        cv2.circle(img, (arrMax,kc_set), 5, (0,0,255),5)
        cv2.line(img, (center,kc_set), (int(img.shape[1]/2),img.shape[0]), (0,255,0),(5))
        cv2.imshow("controlDevice",img)
        print("arrMax", arrMax, "arrMin", arrMin, "center", center,"sendBack_angle", sendBack_angle, "sendBack_Speed", sendBack_Speed)
        peed=700
        p = 2
        i = 1
        d = 1
        mm_max = 500
        mm_min = -500
        if(angle1 > 0):
            if(angle1 < mm_max):
                Control(PID(int((angle1*25)/50),p,i,d),peed)
            else:
                Control(PID(25,p,i,d),peed)
        else:
            if(angle1 > mm_min):
                Control(PID(int((angle1*(-25))/(-50)),p,i,d),peed)
            else:
                Control(PID(-25,p,i,d),peed)
        return img
    def process(self):
        scaleImg = Lane.scaleImg(self, self.img)
        preImg = Lane.preImg(self, scaleImg)
        grayImg = Lane.grayImg(self,preImg)
        grayCTbinImg = Lane.grayCTbinImg(self,scaleImg)
        # GaussianBlur = Lane.GaussianBlur(self,grayImg)
        binaryImg = Lane.binaryImg(self, grayImg)
        cannyEdgeDetection = Lane.cannyEdgeDetection(self,binaryImg)
        houghTransform = Lane.houghTransform(self,cannyEdgeDetection)
        contourImg = Lane.contourImg(self,binaryImg,grayCTbinImg)
        combineImg = Lane.combineImg(self, houghTransform, contourImg)
        regionMasking = Lane.regionMasking(self,combineImg)
        Lane.controlDevice(self,regionMasking)

        cv2.imshow("houghTransform",houghTransform)
        cv2.imshow("contourImg",contourImg)
        cv2.imshow("combineImg",combineImg)

from PIL import Image as im

import main

if __name__ == "__main__":
    # Traffic sign model
    try:
        while True:

            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """
            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                    ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}","utf-8")
            # message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)
            
    

            try:
                image = cv2.imdecode(np.frombuffer(data,np.uint8), -1)
                #-----TRAFFIC SIGN PREDICT-----#

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # detect = main.handle()
                # detect.handle(image)
                cv2.imshow("image", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # image = image.resize((640,360))
                print(image.shape)
                lane = Lane(image)
                lane.process()

            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()
