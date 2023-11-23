import socket
import sys
import time
import cv2
import numpy as np
import json
import base64

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
    def __init__(self) -> None:
        self.height = 360
        self.width = 640
    
    def colorSelection(self,image):
        # Grab the x and y size and make a copy of the image
        ysize = image.shape[0]
        xsize = image.shape[1]
        color_select = np.copy(image)

        # Define color selection criteria
        ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
        red_threshold = 120
        green_threshold = 150
        blue_threshold = 150
        ######

        rgb_threshold = [red_threshold, green_threshold, blue_threshold]

        # Do a boolean or with the "|" character to identify
        # pixels below the thresholds
        thresholds = (image[:,:,0] < rgb_threshold[0]) \
                    | (image[:,:,1] < rgb_threshold[1]) \
                    | (image[:,:,2] < rgb_threshold[2])
        color_select[thresholds] = [0,0,0]

        cv2.imshow("colorSelection", color_select)
        return color_select
    def regionMasking(self,image):
        ysize = image.shape[0]
        xsize = image.shape[1]
        addYSize = 30
        
        # Define the coordinates of the rectangle
        # Format: (x1, y1) - Top-left corner, (x2, y2) - Bottom-right corner
        rectangle_coordinates = [(0, int(ysize/2) + addYSize), (xsize, ysize)]

        # Create a mask with the same size as the image
        mask = np.zeros_like(image)

        # Draw the rectangle on the mask
        cv2.rectangle(mask, rectangle_coordinates[0], rectangle_coordinates[1], (255, 255, 255), thickness=cv2.FILLED)

        # Bitwise AND operation to mask the original image
        result = cv2.bitwise_and(image, mask)
        cv2.imshow("regionMasking",result)
        return result
    def  GaussianBlur(self,image):
        Gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # Define a kernel size for Gaussian smoothing / blurring
        kernel_size = 5 # Must be an odd number (3, 5, 7...)
        GaussianBlur = cv2.GaussianBlur(Gray,(kernel_size, kernel_size),0)
        return GaussianBlur
    def  cannyEdgeDetection(self,image):
        low_threshold = 180
        high_threshold = 240

        GaussianBlur = Lane.GaussianBlur(self, image)
        cannyEdgeDetection = cv2.Canny(GaussianBlur, low_threshold, high_threshold)

        cv2.imshow("cannyEdgeDetection",cannyEdgeDetection)
        return cannyEdgeDetection
    def houghTransform(self, cannyEdgeDetection,img):
        mask = np.zeros_like(cannyEdgeDetection)
        ignore_mask_color = 255
        vertices = np.array([[(0,self.height),(0, int(self.height/2)), (self.width, int(self.height/2)), (self.width,self.height)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(cannyEdgeDetection, mask)
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 20     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 4 #minimum number of pixels making up a line
        max_line_gap = 50    # maximum gap in pixels between connectable line segments
        line_image = np.copy(img)*0 # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((cannyEdgeDetection, cannyEdgeDetection, cannyEdgeDetection)) 

        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
        lines_edges = cv2.polylines(lines_edges,vertices, True, (0,0,255), 10)

        cv2.imshow("lines_edges", lines_edges)
        return lines_edges
    def process(self,img):
        colorSelection = Lane.colorSelection(self, img)
        regionMasking = Lane.regionMasking(self,colorSelection)
        cannyEdgeDetection = Lane.cannyEdgeDetection(self,regionMasking)
        houghTransform = Lane.houghTransform(self,cannyEdgeDetection,img)
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
            cv2.imshow("IMG", imgage)
            # print("Img Shape: ",imgage.shape)
            
            LaneD = Lane()
            LaneD.process(imgage)
            
            
            
            #save image
            # image_name = "./img/img_{}.jpg".format(count)
            # count += 1
            # cv2.imwrite(image_name, imgage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            

    finally:
        print('closing socket')
        s.close()
