import socket
import sys
import time
import cv2
import numpy as np
import json
import base64
import torch

model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') # local model
model1.conf = 0.7

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321

# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 

count = 0
angle = 10
speed = 100

class Lane:
    def __init__(self,img) -> None:
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.img = img
    def preImage(self,img):
        scaleValue = 195  #Scale value x = 200
        img = img[scaleValue:,:]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lower_green = np.array([90,0,0]).astype("uint8")
        upper_green = np.array([140,255,255]).astype("uint8")
        mask = cv2.inRange(img,lower_green,upper_green)
        lane = cv2.bitwise_and(img, img, mask=mask)
        lane = cv2.cvtColor(lane,cv2.COLOR_HSV2BGR)
        color_select = np.copy(lane)
        red_threshold = 8
        green_threshold = 8
        blue_threshold = 8
        rgb_threshold = [red_threshold, green_threshold, blue_threshold]
        thresholds = (img[:,:,0] < rgb_threshold[0]) \
                    | (img[:,:,1] < rgb_threshold[1]) \
                    | (img[:,:,2] < rgb_threshold[2])
        color_select[thresholds] = [255,255,255]
        color_select = cv2.cvtColor(color_select,cv2.COLOR_BGR2GRAY)
        threshold = 40
        _, binary_thresh = cv2.threshold(color_select, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("lane",lane)
        cv2.imshow("color_select",color_select)
        cv2.imshow("binary_thresh",binary_thresh)
        return binary_thresh
    def detectSign(self,image):
        imageOD = cv2.resize(image, (320, 320))
        # cv2.imshow("imageOD", imageOD)
        results = model1(imageOD)
        # print(len(results.pandas().xyxy[0].confidence))
        if (len(results.pandas().xyxy[0]) != 0):
            if (float(results.pandas().xyxy[0].confidence[0])) >= 0.85:

                x_min = int(results.xyxy[0][0][0])
                y_min = int(results.xyxy[0][0][1])
                x_max = int(results.xyxy[0][0][2])
                y_max = int(results.xyxy[0][0][3])

                x_c = int(x_min + (x_max - x_min)/2)
                y_c = int(y_min + (y_max - y_min)/2)

                s_bbox = (x_max - x_min) * (y_max - y_min)
                
                
                
                img_classifier = imageOD[y_min:y_max, x_min:x_max]
                cv2.imshow("abc",img_classifier)
                sign = Predict(img_classifier)

                if s_bbox > 30 and s_bbox < 600:
                    if results.pandas().xyxy[0].name[0] == 'unknown' or sign == "unknown":
                        # print("Normal", s_bbox)
                        return "none"
                    else:
                        # print("Slow down", s_bbox)
                        return "decrease"


                elif s_bbox >= 600 and s_bbox <= 1800 and y_min > 10 and float(results.pandas().xyxy[0].confidence[0]) > 0.88:  # and y_min > 10 and x_max < 270:
                    cv2.rectangle(imageOD, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
                    # cv2.imshow("cls", img_classifier)
                    if sign != "unknown":
                        sign_checked = check_sign(sign, 2)
                        if sign_checked != "none":
                            return sign_checked
                        else:
                            return "unknown"
                    else:
                        # print("Unknown ------ Ignore")
                        return "unknown"

            else:
                return "none"
        else:
            signArray[1:] = signArray[0:-1]
            signArray[0] = 0
            return "none"
    def process(self):
        cv2.imshow("image ban dau", self.img)
        preImage = Lane.preImage(self,self.img)
        return

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
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            # print(data)
            data_recv = json.loads(data)

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            # print("speed: ", current_speed)
            # print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            imgage = cv2.imdecode(jpg_as_np, flags=1)
            # cv2.imshow("IMG", imgage)
            # print("Img Shape: ",imgage.shape)

            lane = Lane(imgage)
            lane.process()
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