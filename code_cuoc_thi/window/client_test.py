import socket
import sys
import time
import cv2
import numpy as np
import json
import base64
import math

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321

# connect to the server on local computer 
s.connect(('127.0.0.1', port))

# count = 0
angle = 10
speed = 100

global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed

#-----------------PID Controller-------------------#
error_arr = np.zeros(5)
pre_t = time.time()
MAX_SPEED = 60

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
        scaleValue = 195 #Scale value x
        return img[scaleValue:,:]
    def grayImg(self, img):
        lower_green = np.array([95,0,0]).astype("uint8")
        upper_green = np.array([255,255,255]).astype("uint8")

        mask = cv2.inRange(img,lower_green,upper_green)
        lane = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(lane,cv2.COLOR_BGR2GRAY)
    def contourImg(self,img):
        cv2.imshow("iiiiii",img)
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
        cv2.drawContours(img,[cnt],0,(255,0,255),cv2.FILLED)
        _, binary_thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        return binary_thresh
    def colorSelection(self, img):
        color_select = np.copy(img)
        red_threshold = 100
        green_threshold = 140
        blue_threshold = 100
        rgb_threshold = [red_threshold, green_threshold, blue_threshold]
        thresholds = (img[:,:,0] < rgb_threshold[0]) \
                    | (img[:,:,1] < rgb_threshold[1]) \
                    | (img[:,:,2] < rgb_threshold[2])
        color_select[thresholds] = [0,0,0]
        cv2.imshow("colorSelection", color_select)
        return color_select
    def regionMasking(self, img):
        addYSize = 30
        rectangle_coordinates = [(0, int(self.height/2) + addYSize), (self.width, self.height)]
        mask = np.zeros_like(img)
        cv2.rectangle(mask, rectangle_coordinates[0], rectangle_coordinates[1], (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(img, mask)
        cv2.imshow("regionMasking",result)
        return result
    def  GaussianBlur(self,img):
        Gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        kernel_size = 3
        GaussianBlur = cv2.GaussianBlur(Gray,(kernel_size, kernel_size),0)
        return GaussianBlur
    def  cannyEdgeDetection(self,img):
        low_threshold = 150
        high_threshold = 255
        GaussianBlur = Lane.GaussianBlur(self, img)
        cannyEdgeDetection = cv2.Canny(GaussianBlur, low_threshold, high_threshold)
        cv2.imshow("cannyEdgeDetection",cannyEdgeDetection)
        return cannyEdgeDetection
    def houghTransform(self, cannyEdgeDetection):
        _img_bgr = np.copy(Lane.scaleImg(self, self.img))
        rho = 1              # distance resolution in pixels of the Hough grid
        theta = np.pi/180    # angular resolution in radians of the Hough grid
        threshold = 20       # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 4  # minimum number of pixels making up a line
        max_line_gap = 50    # maximum gap in pixels between connectable line segments
        lineP = cv2.HoughLinesP(cannyEdgeDetection, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
        if lineP is not None:
            for i in range(0, len(lineP)):
                l = lineP[i][0]
                cv2.line(_img_bgr,(l[0],l[1]),(l[2],l[3]),(255,255,255),8,cv2.LINE_AA)
        _, binary_thresh = cv2.threshold(_img_bgr, 220, 255, cv2.THRESH_BINARY)
        return binary_thresh
    def controlDevice(self,img1,img2):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        bitwiseOr = cv2.bitwise_or(img1, img2)
        contours, _ = cv2.findContours(bitwiseOr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(bitwiseOr)
        cv2.imshow("mask",bitwiseOr)
        cv2.drawContours(mask, [largest_contour], 0, (255), thickness=cv2.FILLED)
        result = np.ones_like(bitwiseOr) * 255
        result = cv2.bitwise_and(result, result, mask=mask)
        dd_hold = 82
        lineRow = result[dd_hold,:]
        arr=[]
        for x,y in enumerate(lineRow):
            if y == 255:
                arr.append(x)
        d_set = 300
        arrMax = max(arr)
        arrMin = min(arr)
        # print(arrMax-arrMin)
        center = int((arrMax + arrMin)/2)
        angle1 = math.degrees(math.atan((center - result.shape[1]/2)/(result.shape[0]-dd_hold)))
        # print("angle",angle)
        cv2.circle(result, (arrMin,dd_hold), 5, (0,0,255),5)
        cv2.circle(result, (arrMax,dd_hold), 5, (0,0,255),5)
        cv2.line(result, (center,dd_hold), (int(result.shape[1]/2),result.shape[0]), (0,0,255),(5))
        # print("center",center)
        mm_max = 500
        mm_min = -500
        p = 20   #20
        i = 11   #11
        d = 1    #1
        peed=150
        if(arrMax-arrMin > d_set):
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
        else:
            Control(0,100)
        cv2.imshow("lineRow",lineRow)
        cv2.imshow("controlDevice",result)
        return img1
    def process(self):
        scaleImg = Lane.scaleImg(self, self.img)
        grayImg = Lane.grayImg(self, scaleImg)
        contourImg = Lane.contourImg(self, grayImg)
        colorSelection = Lane.colorSelection(self,scaleImg)
        cannyEdgeDetection = Lane.cannyEdgeDetection(self, colorSelection)
        houghTransform = Lane.houghTransform(self, cannyEdgeDetection)
        controlDevice = Lane.controlDevice(self, contourImg, houghTransform)
        cv2.imshow("contourImg",contourImg)
        # cv2.imshow("cannyEdgeDetection",cannyEdgeDetection)
        cv2.imshow("houghTransform",houghTransform)

if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            message = bytes(f"{sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)
            
            data_recv = json.loads(data)
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            cv2.imshow("IMG", image)
            
            lane = Lane(image)
            lane.process()
            print(sendBack_angle, sendBack_Speed)
            #save image
            # image_name = "./img/img_{}.jpg".format(count)
            # image_name = "E://WORKPACE//XE_TU_HANH//labelme//image//img_{}.jpg".format(count)
            # count += 1
            # cv2.imwrite(image_name, imgage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            

    finally:
        print('closing socket')
        s.close()
