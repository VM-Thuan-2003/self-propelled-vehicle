import socket
import sys
import time
import cv2
import numpy as np
import json
import base64
import torch
import math

max = 0
model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:\WORKPACE\XE_TU_HANH\code\RealTime_Detection-And-Classification-of-TrafficSigns-main\Model\weights\\best.pt') # local model
model.conf = 0.7
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(("127.0.0.1", port))

global angle, speed
count = 0
angle = 0
speed = 150


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
           
            # Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)
            
            #--------------------DETECT---------------------#
            img1 = img[..., ::-1]
            results = model([img1])
            data = results.pandas().xyxy[0]
            temp = str(data["xmin"].values)
            print(data)
            if(temp != '[]'):
                temp = 0
                max = 0
                flag = 0
                for i in data["confidence"].values:
                    if(flag == 0):
                        temp = i
                        flag+=1
                    elif(temp < i):
                        max+=1
                
                #coordinates
                x1 = int(data["xmin"].values[max])
                y1 = int(data["ymin"].values[max])
                x2 = int(data["xmax"].values[max])
                y2 = int(data["ymax"].values[max])
                
                #confidence
                conf = data["confidence"].values[max]
                
                #object details
                cv2.putText(img,text=data["name"].values[max],org=[x1-50,y2-50],thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0))
                
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                
            #------------------------------------------------#
            cv2.imshow("IMG", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    finally:
        print("closing socket")
        s.close()
