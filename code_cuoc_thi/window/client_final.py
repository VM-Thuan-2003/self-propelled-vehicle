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
        lower_green = np.array([110,0,0]).astype("uint8")
        upper_green = np.array([255,255,255]).astype("uint8")

        mask = cv2.inRange(img,lower_green,upper_green)
        lane = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(lane,cv2.COLOR_BGR2GRAY)
    def preNoneCenter(self,img):
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
        
        edges = cv2.Canny(img, 50, 150)
        # Tìm đường biên trong hình ảnh
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Tìm cạnh ở bên trái và bên phải
        left_edges = []
        right_edges = []

        for contour in contours:
            # Xác định tọa độ của hình chữ nhật bao quanh đường biên
            x, y, w, h = cv2.boundingRect(contour)

            # Chia đường biên thành hai phần, xác định xem nó thuộc bên trái hay bên phải
            mid_x = x + w // 2

            if mid_x < self.width // 2:
                left_edges.append(contour)
            else:
                right_edges.append(contour)

        # Vẽ cạnh bên trái và bên phải lên hình ảnh gốc
        image_with_edges = cv2.drawContours(img.copy(), left_edges, -1, (255, 0, 0), 2)
        image_with_edges = cv2.drawContours(image_with_edges, right_edges, -1, (0, 0, 255), 2)
        cv2.imshow('Edges on Left and Right', image_with_edges)
        return img
    def contourImg(self,img):
        img = Lane.preNoneCenter(self,img)
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
        peed=500
        p = 17
        i = 11
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
        contourImg = Lane.contourImg(self,binaryImg)
        combineImg = Lane.combineImg(self, houghTransform, contourImg)
        regionMasking = Lane.regionMasking(self,combineImg)
        Lane.controlDevice(self,regionMasking)
        
        cv2.imshow("houghTransform",houghTransform)
        cv2.imshow("contourImg",contourImg)
        # cv2.imshow("combineImg",combineImg)
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
            
            # print(sendBack_angle, sendBack_Speed)
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
