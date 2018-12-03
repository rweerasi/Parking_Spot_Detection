import numpy as np
import cv2

img = cv2.imread('parking_new_10.jpg')
#cv2.imshow('img2',img)
#cv2.waitKey(0)
#m,n = img.shape
#img = cv2.GaussianBlur(img,(5,5),0)

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img_grey,11)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,0)
th2 = cv2.medianBlur(th2,7)
#th2 = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
print(th2)
m,n = th2.shape
#th2= cv2.bitwise_not(th2)
cv2.imshow('thresh', th2)
cv2.waitKey(5000)
cv2.imwrite('thresh.png', th2)
#cv2.destroyAllWindows()
