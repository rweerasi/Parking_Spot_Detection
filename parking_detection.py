import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.misc import imsave
import os
import city_segment


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
    cannyed_image = cv2.Canny(gray_image, 50, 250) #first we get the canny edge detected linesw
    imsave("cannyed_image.png", cannyed_image)
    if type(roi) == type([]):
        cropped_image = mask_image(cannyed_image, region_of_interest(cannyed_image, np.array([roi],np.int32)))
    if type(roi) == type(np.array([])):
        cropped_image = mask_image(cannyed_image, roi)
    lines = cv2.HoughLinesP(cropped_image,rho=rho,theta=angle,threshold=thresh,lines=np.array([]),minLineLength=min_length,maxLineGap=max_gap)
    #print(lines)
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

def get_slopes(img,lines):

    lines = np.array(lines)
    x_vec = lines[:,:,0] - lines[:,:,2]
    y_vec = lines[:,:,1] - lines[:,:,3]
    slope = np.divide(y_vec,x_vec)
    intercept = lines[:,:,1] - np.multiply(lines[:,:,0],slope)
    print(slope)
    print(intercept)
    n_lines, _ = np.shape(intercept)

    new_lines = np.zeros((n_lines*n_lines,1,4))
    k = 0


    for i in range(0,n_lines):
        for j in range(i, n_lines):

            if (abs(slope[i,0] - slope[j,0]) < 0.1 and abs(intercept[i,0] - intercept[j,0]) < 2):
                if (lines[i,0,0] > lines[j,0,0]):

                    new_lines[k,0,0] = lines[j,0,0]
                    new_lines[k,0,1] = lines[j,0,1]
                    new_lines[k,0,2] = lines[i,0,2]
                    new_lines[k,0,3] = lines[i,0,3]

                else:

                    new_lines[k,0,0] = lines[i,0,0]
                    new_lines[k,0,1] = lines[i,0,1]
                    new_lines[k,0,2] = lines[j,0,2]
                    new_lines[k,0,3] = lines[j,0,3]

            k = k+1

    new_lines = new_lines[:k,:,:]
    new_lines = new_lines.astype(int)

    return new_lines

def lines_processing(lines):
    """ Processes the line from the Hough Transform in order to glean
        information about parking spots
    Input:
        lines:              list of lines, represented as 4 values, (start_x, 
                            start_y, end_x, end_y)
    Outputs:
        processed_lines:    list of lines, in the same format as input, which 
                            better represents the parking spots present in
                            the image
    """
    long_lines = []
    for line in lines:
        line = line[0]
        length = np.sqrt((line[0] - line[2])**2 + (line[1] - line[3])**2)
        if length > 8:
            long_lines.append([line])

    horizontal_lines = []
    for line in long_lines:
        line = line[0]
        angle = np.arctan2(line[3] - line[1], line[2] - line[0])
        # Get only horizontal(ish) lines
        max_ang = 0.25
        if (angle < max_ang and angle > -max_ang) or \
            (angle > np.pi - max_ang or angle < -np.pi + max_ang):
            horizontal_lines.append([line])

    return horizontal_lines
    '''
    left_lines = []; right_lines = []
    for line in horizontal_lines:
        line = line[0]
        if line[0] < 1024 and line[2] < 1024:
            left_lines.append([line])
        if line[0] > 1024 and line[2] > 1024:
            right_lines.append([line])
    ''' 

    #return left_lines, right_lines


if __name__ == "__main__":

    #img=cv2.imread("parking_2.png")
    #height, width, channels = img.shape
    param=[6,0.01,100,30,35]
    filen="example.jpg"

    """
    rho=rho,theta=angle,threshold=thresh,lines=np.array([]),minLineLength=min_length,maxLineGap=max_gap)
    """
    if not os.path.isfile("output.npy"):
        city_segment.segmentation(filen, "output.jpg")

    segments = np.load("output.npy")
    mask = 255 * (segments == 0).astype(np.uint8)

    mask_road = np.zeros(mask.shape)
    mask_road[segments == 0] = 255
    img = cv2.imread(filen)
    mask = np.dstack((mask_road, mask_road, mask_road)).astype(np.uint8)
    masked_image = mask_image(img, mask)


    # Apply traditional CV to masked image
    lines = get_lines(img,  mask_road.astype(np.uint8),param)
    h_lines = lines_processing(lines)
    #l_new_lines, _ = lines_processing(get_slopes(img,l_lines))
    new_lines = get_slopes(img,h_lines)
    line_image = draw_lines(img, h_lines)
    line_image_slope = draw_lines(img, new_lines)
    imsave("woslope.png",line_image)
    
    imsave("wslope.png",line_image_slope)
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
