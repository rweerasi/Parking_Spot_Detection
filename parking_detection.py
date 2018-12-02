import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.misc import imsave
import os
import city_segment
import math


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # making grayscale
    cv2.fillPoly(mask, vertices, match_mask_color) #filling the polygon
    return mask


def mask_image(img, mask):
    return cv2.bitwise_and(img, mask)

def get_lines(img, roi, param=[6,20,80,20,35]):
    """
    Lines is the variable that will store the coordinates of all lines
    detected using Hough Transform
    Tweaking these parameters is a challenge in the HoughLinesP function.
    Optional: Can we make this dynamic?
    """
    rho=param[0]
    angle=param[1]*np.pi/180
    thresh=param[2]
    min_length=param[3]
    max_gap=param[4]
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 10, 60) #first we get the canny edge detected linesw
    imsave("cannyed_image.png", cannyed_image)
    if type(roi) == type([]):
        cropped_image = mask_image(cannyed_image, region_of_interest(cannyed_image, np.array([roi],np.int32)))
    if type(roi) == type(np.array([])):
        cropped_image = mask_image(cannyed_image, roi)
    lines = cv2.HoughLinesP(cropped_image,rho=rho,theta=angle,threshold=thresh,lines=np.array([]),minLineLength=min_length,maxLineGap=max_gap)
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
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = img.shape[0] * (3 / 5) # <-- Just below the horizon
    max_y = img.shape[0] # <-- The bottom of the image
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    line_image = draw_lines(
        img,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )
    plt.figure()
    plt.imshow(line_image)
    plt.show()

    return img



if __name__ == "__main__":

    #img=cv2.imread("parking_2.png")
    #height, width, channels = img.shape
    param=[6,0.01,100,50,35]
    filen="parking_5.png"

    """
    rho=rho,theta=angle,threshold=thresh,lines=np.array([]),minLineLength=min_length,maxLineGap=max_gap)
    """
    if not os.path.isfile("output.npy"):
        city_segment.segmentation(filen, "output.png")

    segments = np.load("output.npy")
    mask = 255 * (segments == 0).astype(np.uint8)

    mask_road = np.zeros(mask.shape)
    mask_road[segments == 0] = 255
    img = cv2.imread(filen)
    mask = np.dstack((mask_road, mask_road, mask_road)).astype(np.uint8)
    masked_image = mask_image(img, mask)


    # Apply traditional CV to masked image
    lines = get_lines(img,  mask_road.astype(np.uint8),param)
    line_image = draw_lines(img, lines)

    # Save images
    imsave("mask_image.png",masked_image)

    imsave("line_image"+str(param)+".png", line_image)
    plt.figure()
    plt.imshow(line_image)
    plt.show()

    #region_of_interest_vertices = [(0, height),(0, height/2.5),(width, height/2.5),]

    #cropped_image = mask_image(img, region_of_interest(img,np.array([region_of_interest_vertices], np.int32),)) #the actual cropped image
    #plt.figure()
    #plt.imshow(cropped_image)
    #plt.show()
    #imsave("cropped_image.png", cropped_image)

    #lines = get_lines(img, region_of_interest_vertices)
    #line_image = draw_lines(img, lines)
    #imsave("line_image.png", line_image)
    #plt.figure()
    ##plt.imshow(line_image)
    #plt.show()
