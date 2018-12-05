import os 
import numpy as np
import cv2
import city_segment

if __name__ == "__main__":
    img = cv2.imread("example.jpg")

    # Preprocessing for without shadow
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    blur1 = cv2.medianBlur(gray1, 11)
    ret,th3shadow = cv2.threshold(blur1,100,255,cv2.THRESH_BINARY)
    th3 = cv2.medianBlur(th3shadow, 5)

    # Canny edge detector
    cannyed_image_shadow = cv2.Canny(th3, 10, 250)

    # Preprocessing for canny with shadow
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    blur = cv2.medianBlur(gray, 5)
        
    cannyed_image = cv2.Canny(blur, 75, 250)

    # Road only mask
    if not os.path.isfile("output.npy"):
        city_segment.segmentation("example.jpg", "output.jpg")
    mask = city_segment.largest_connected_component()

    # Apply mask 
    canny_road1 = cv2.bitwise_and(cannyed_image_shadow, mask)
    canny_road2 = cv2.bitwise_and(cannyed_image, mask)

    # Subtraction 
    woshadow = canny_road2 - canny_road1
    
    # Show 
    cv2.imshow("", woshadow)
    cv2.waitKey(-1)
