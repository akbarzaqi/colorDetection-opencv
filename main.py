import cv2
import numpy as np
import matplotlib.pyplot as plt

root_path = "img/1.jpg"

def read_image(image_path):
 bgr_img = cv2.imread(image_path)
 rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
 return rgb_img

def show_image(rgb_img):
 cv2.imshow("Original Image", rgb_img)
 
def trackbar_gawang():
    cv2.createTrackbar("H_MIN", "Gawang", 0, 179, nothing)
    cv2.createTrackbar("H_MAX", "Gawang", 179, 179, nothing)
    cv2.createTrackbar("S_MIN", "Gawang", 0, 255, nothing)
    cv2.createTrackbar("S_MAX", "Gawang", 255, 255, nothing)
    cv2.createTrackbar("V_MIN", "Gawang", 0, 255, nothing)
    cv2.createTrackbar("V_MAX", "Gawang", 255, 255, nothing)

def nothing(x):
    pass

cv2.namedWindow("Gawang")
trackbar_gawang()
 
jpg_img = read_image(root_path)
resize_img = cv2.resize(jpg_img, (320, 240))

hsv_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2HSV)


while True:
   
    
    h_minGawang = cv2.getTrackbarPos("H_MIN", "Gawang")
    h_maxGawang = cv2.getTrackbarPos("H_MAX", "Gawang")
    s_minGawang = cv2.getTrackbarPos("S_MIN", "Gawang")
    s_maxGawang = cv2.getTrackbarPos("S_MAX", "Gawang")
    v_minGawang = cv2.getTrackbarPos("V_MIN", "Gawang")
    v_maxGawang = cv2.getTrackbarPos("V_MAX", "Gawang")
    
    lower_yellow = np.array([h_minGawang, s_minGawang, v_minGawang])
    upper_yellow = np.array([h_maxGawang, s_maxGawang, v_maxGawang])
    
    maskGawang = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    resultGawang = cv2.bitwise_and(resize_img, resize_img, mask = maskGawang)
    
    copy_img = resize_img.copy()
    
    contours, _ = cv2.findContours(maskGawang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(copy_img, (x,y), (x + w, y + h), (0, 255, 0), 2)
                text_position = (x, y - 10)  
                cv2.putText(copy_img, "Gawang", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("mask gawang", maskGawang)
    cv2.imshow("result gawang", resultGawang)
    show_image(copy_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()