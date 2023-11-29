import cv2
import numpy as np

class Lane:
    def __init__(self,img) -> None:
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.img = img
    
    def scaleImg(self,img):
        scaleValue = 190 #Scale value x
        return img[scaleValue:,:]
    def grayImg(self, img):
        lower_green = np.array([95,0,0]).astype("uint8")
        upper_green = np.array([255,255,255]).astype("uint8")

        mask = cv2.inRange(img,lower_green,upper_green)
        lane = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(lane,cv2.COLOR_BGR2GRAY)
    def contourImg(self,img):
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
        cv2.drawContours(img,[cnt],0,(255,0,255),5)
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
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 20     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 4 #minimum number of pixels making up a line
        max_line_gap = 50    # maximum gap in pixels between connectable line segments
        lineP = cv2.HoughLinesP(cannyEdgeDetection, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
        if lineP is not None:
            for i in range(0, len(lineP)):
                l = lineP[i][0]
                cv2.line(_img_bgr,(l[0],l[1]),(l[2],l[3]),(255,255,255),8,cv2.LINE_AA)
        _, binary_thresh = cv2.threshold(_img_bgr, 220, 255, cv2.THRESH_BINARY)
        return binary_thresh
    def process(self):
        scaleImg = Lane.scaleImg(self, self.img)
        grayImg = Lane.grayImg(self, scaleImg)
        contourImg = Lane.contourImg(self, grayImg)
        colorSelection = Lane.colorSelection(self,scaleImg)
        cannyEdgeDetection = Lane.cannyEdgeDetection(self, colorSelection)
        houghTransform = Lane.houghTransform(self, cannyEdgeDetection)
        cv2.imshow("gray",contourImg)
        cv2.imshow("cannyEdgeDetection",cannyEdgeDetection)
        cv2.imshow("houghTransform",houghTransform)
        cv2.waitKey(0)


if __name__ == __name__:
    path = "./imgs/lane_treeShadow.jpg"
    img_bgr = cv2.imread(path)
    lane = Lane(img_bgr)
    lane.process()