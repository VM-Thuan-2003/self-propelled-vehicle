import cv2
import numpy as np

def main():
    path = "./imgs/lane_treeShadow.jpg"
    img_bgr = cv2.imread(path)
    img_bgr = img_bgr[190:,:]
    
    lower_green = np.array([95,0,0]).astype("uint8")
    upper_green = np.array([255,255,255]).astype("uint8")

    mask = cv2.inRange(img_bgr,lower_green,upper_green)
    lane = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    cv2.imshow("llll",lane)
    lane_gray = cv2.cvtColor(lane,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(lane_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    area=[]
    
    for i, contour in enumerate(contours):
        area.append(cv2.contourArea(contour))

    max_area = np.max(area)
    idx = area.index(max_area)
    
    print(idx)
    cnt = contours[idx]
    print(cnt)
    cv2.drawContours(lane,[cnt],0,(255,0,255),1)
    
    cv2.imshow("lane",lane)
    cv2.imshow("lane_gray",lane_gray)
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    
if(__name__ == __name__):
    main()