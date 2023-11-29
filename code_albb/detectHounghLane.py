import cv2
import numpy as np

def main():
    path = "./imgs/lane_treeShadow.jpg"
    img_bgr = cv2.imread(path)
    img_bgr = img_bgr[190:,:]
    
    gray_img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(1,1),0)
    canny = cv2.Canny(blur_img,150,255)
    
    _img_bgr = np.copy(img_bgr)
    lineP = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)
    if lineP is not None:
        for i in range(0, len(lineP)):
            l = lineP[i][0]
            cv2.line(_img_bgr,(l[0],l[1]),(l[2],l[3]),(0,0,255),1,cv2.LINE_AA)
            
    cv2.imshow("lane",img_bgr)
    cv2.imshow("_img_lane",_img_bgr)
    cv2.imshow("canny",canny)
    cv2.waitKey(0)
if(__name__ == __name__):
    main()