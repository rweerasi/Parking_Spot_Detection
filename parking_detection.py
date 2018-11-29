import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.misc import imsave



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # making grayscale
    cv2.fillPoly(mask, vertices, match_mask_color) #filling the polygon
    return mask


def mask_image(img, mask):
    return cv2.bitwise_and(img, mask)

def get_lines(img, roi_vertices): 
    """
    Lines is the variable that will store the coordinates of all lines 
    detected using Hough Transform 
    Tweaking these parameters is a challenge in the HoughLinesP function. 
    Optional: Can we make this dynamic?
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 200, 300) #first we get the canny edge detected linesw
    cropped_image = mask_image(cannyed_image, region_of_interest(cannyed_image, np.array([roi_vertices],np.int32)))#the ROI is then chosen using canny edge detection
    lines = cv2.HoughLinesP(cropped_image,rho=6,theta=np.pi / 60,threshold=120,lines=np.array([]),minLineLength=20,maxLineGap=15)
    return lines


"""
following is a function which draws the lines on the image. I have used it as 
is from the page
https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
"""
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8,)
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img



if __name__ == "__main__":
    
    img=cv2.imread("example.png") 
    height, width, channels = img.shape 

    region_of_interest_vertices = [(0, height),(0, height/2.5),(width, height/2.5),] 

    cropped_image = mask_image(img, region_of_interest(img,np.array([region_of_interest_vertices], np.int32),)) #the actual cropped image
    plt.figure()
    plt.imshow(cropped_image)
    plt.show()
    imsave("cropped_image.png", cropped_image)

    lines = get_lines(img, region_of_interest_vertices)
    line_image = draw_lines(img, lines)
    imsave("line_image.png", line_image)
    plt.figure()
    plt.imshow(line_image)
    plt.show()


