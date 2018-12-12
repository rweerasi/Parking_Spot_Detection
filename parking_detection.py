import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.misc import imsave
import os
import city_segment
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch



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
    kernel_size = 5
    gray_image = cv2.GaussianBlur(gray_image,(kernel_size, kernel_size),0)
    cannyed_image = cv2.Canny(gray_image, 50, 150) #first we get the canny edge detected linesw
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

    return img

def line_dist(line1, line2):

    return np.linalg.norm(line1-line2)


def get_scatter(lines):

    """
    This function gives us the points to a scatter plot where
    the points in the plot are points sampled from the line made by
    Hough Transform

    INPUT: The lines which are returned by the Hough Transform

    OUTPUT: The scatter plot corresponding to those lines. We also have outputted
        a .csv file which saves these points

    """


    lines = np.array(lines)
    x1_vec = lines[:,0,0]  
    y1_vec = lines[:,0,1] 
    x2_vec = lines[:,0,2]
    y2_vec = lines[:,0,3]

    x1_vec = x1_vec.astype(int)
    x2_vec = x2_vec.astype(int)
    y1_vec = y1_vec.astype(int)
    y2_vec = y2_vec.astype(int)

    x_points = np.array([])
    x_points = x_points.astype(int)
    y_points = []

    a = np.shape(x1_vec)
    a = a[0]

    for i in range(0,a):
        x_points = np.append(x_points,np.linspace(x1_vec[i],x2_vec[i],10))
        y_points = np.append(y_points,np.linspace(y1_vec[i],y2_vec[i],10))

    X = np.vstack((x_points,y_points))

    plt.scatter(x_points,y_points)
    np.savetxt("foo.csv", X, delimiter=",")

    plt.savefig("scatter.png")

def hc():

    """

    This function takes the points from a .csv file and performs hierarchical clustering.
    You have to look the dendogram that is displayed and chosoe the number of clusters 
    accordingly

    INPUT: This function takes the scatter plot from the .csv files and performs 
    hierarchical clustering over it

    PARAMETERS: You may have to tune to number of clusters based on the parking
    lot image you have.

    OUTPUT: The scatter plot and the computed cluster labels are then returned.
    """

    X = np.genfromtxt('foo.csv',delimiter=',')

    X = np.transpose(X)

    X[:,1] = 10*X[:,1]

    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.ylabel('Skewed Euclidean distances')
    plt.savefig("dendogram.png")

    #print(sch.linkage(X, method = 'ward')[-30:,2])


    n_clust = 9
    # Fitting Hierarchical Clustering to the dataset
    hc = AgglomerativeClustering(n_clusters = n_clust, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)

    X[:,1] = np.round(0.1*X[:,1])

    return X, y_hc, n_clust

def parking_spots(X,y_hc,n_clust):

    """
    The following function takes the scatter plot and the cluster labels. It then joins the
    furthest points in the clsuter labels so as to form exactly one continuous line per parking
    line. We now have a one to one correspondance between number of parking lines and the lines
    in the figure

    INPUTS:

        X: The scatter plot made using the original Hough Lines

        y_hc: The cluster labels

        n_clust: Number of clusters

    OUTPUTS:

        lines: The line which is formed my connecting the farthest points in the cluster

    """


    lines = np.zeros((n_clust,1,4))
    for i in range(n_clust):

        k = np.argmin(X[y_hc == i, 0]) 
        lines[i,0,0] = np.amin(X[y_hc == i, 0])
        lines[i,0,1] = X[y_hc == i, 1][k]

        k = np.argmax(X[y_hc == i, 0]) 
        lines[i,0,2] = np.amax(X[y_hc == i, 0])
        lines[i,0,3] = X[y_hc == i, 1][k]

        lines = lines.astype(int)

    return lines
        


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
        max_ang = 0.15
        if (angle < max_ang and angle > -max_ang) or \
            (angle > np.pi - max_ang or angle < -np.pi + max_ang):
            horizontal_lines.append([line])

    return horizontal_lines


def extend_lines(lines):

    """
    The following functions extends the lines until the end of the image. This is
    necessary because the Hough Transform usually doesn't detect
    the entire line. The lines in the left are extended till x = 0 and the lines
    on the right are extended till x = 2047

    INPUT:
    lines: list of lines, represented as 4 values, (start_x, start_y, end_x, end_y)

    OUTPUT:
    long_lines: list of extended lines, in the same format as the input

    """

    border = 700
    left_lines = []; right_lines = []
    for line in lines:
        line = line[0]
        if line[0] < border and line[2] < border:
            left_lines.append([line])
        if line[0] > border and line[2] > border:
            right_lines.append([line])

    # Extend to edge of screen
    long_lines = []
    for line in left_lines:
        line = line[0]
        if line[0] < line[2]:
            x1 = line[0]; y1 = line[1]
            x2 = line[2]; y2 = line[3]
        else:
            x1 = line[2]; y1 = line[3]
            x2 = line[0]; y2 = line[1]
        m = (y2-y1)/(x2-x1)
        b = y1 - m * x1
        long_lines.append([[0, int(b), x2, y2]])
    for line in right_lines:
        line = line[0]
        if line[0] < line[2]:
            x1 = line[0]; y1 = line[1]
            x2 = line[2]; y2 = line[3]
        else:
            x1 = line[2]; y1 = line[3]
            x2 = line[0]; y2 = line[1]
        m = (y2-y1)/(x2-x1)
        b = y1 - m * x1
        long_lines.append([[x1, y1, 2046, int(m*2046+b)]])

    return long_lines

def highlight_spots(img, long_lines, car_lines):

    """
    This function creates the bounding box of the parking spot. Then it checks
    if the parking spot is occupied by a car or not. If the spot it occupied, it
    is highlighted to be red, if it is vacant, it is highlighted to be green

    INPUTS:

        img: The given image from which the parking spot is to be detected

        long_lines: The extended lines from the hough transform and clustering
            represented as 4 values, (start_x, start_y, end_x, end_y)

        car_lines: The line of the bottom of the bounding box of the car in the
            same format as long_lines

    OUTPUTS: The image with the highlighted parking spots is displayed and saved.

    """

    # Separate into left and right 
    border = 700
    left_lines = []; right_lines = []
    for line in long_lines:
        line = line[0]
        if line[0] < border and line[2] < border:
            left_lines.append([line])
        if line[0] > border and line[2] > border:
            right_lines.append([line])

    # Sort left lines by y-intercept
    keys = []
    for line in left_lines:
        line = line[0]
        keys.append(line[3])
    order = np.argsort(keys)
    ordered_lines = []
    for ii in range(len(left_lines)):
        ordered_lines.append(left_lines[order[ii]])

    ncar = len(car_lines)

    # Draw rectangles 
    for ii in range(len(left_lines) - 1):
        a = np.array(ordered_lines[ii]).reshape((2,2))
        b = np.flipud(np.array(ordered_lines[ii+1]).reshape((2,2)))
        pts = np.vstack((a, b))
        flag = 0

        for jj in range(ncar):

            mean_x = 0.5*(car_lines[jj][0][0] + car_lines[jj][0][2])
            mean_y = 0.5*(car_lines[jj][0][1] + car_lines[jj][0][3])

            x_least = 0
            #x_high = max(a[1][0],b[0][0])
            y_least = a[0][1]
            #y_high = min(a[1][1],a[0][1])
            y_high = b[1][1]
            x_high = a[1][0]

            print(a)
            print(b)
            print("Left\n")

            if car_lines[jj][0][2]  > x_least and car_lines[jj][0][2]  < x_high and \
            car_lines[jj][0][3] > y_least and car_lines[jj][0][3] < y_high:
                print(car_lines[jj][0][1])
                flag = 1

 
        if flag == 1:
            cv2.fillPoly(img, [pts], (0,0,255))
            #cv2.polylines(img, [pts], True, (0,0,255), 3)
        else:
            cv2.fillPoly(img, [pts], (0,255,0))

    # Sort right by right-intercept
    keys = []
    for line in right_lines:
        line = line[0]
        keys.append(line[1])
    order = np.argsort(keys)
    ordered_lines = []
    for ii in range(len(right_lines)):
        ordered_lines.append(right_lines[order[ii]])

    # Draw rectangles
    for ii in range(len(right_lines) - 1):
        a = np.array(ordered_lines[ii]).reshape((2,2))
        b = np.flipud(np.array(ordered_lines[ii+1]).reshape((2,2)))
        pts = np.vstack((a,b))
        flag = 0
        for jj in range(ncar):

            mean_x = 0.5*(car_lines[jj][0][0] + car_lines[jj][0][2])
            mean_y = 0.5*(car_lines[jj][0][1] + car_lines[jj][0][3])

            x_least = a[0][0]
            #x_high = max(a[1][0],b[0][0])
            y_least = a[0][1]
            #y_high = min(a[1][1],a[0][1])
            y_high = b[1][1]
            x_high = 2046

            print(a)
            print(b)
            print("Right\n")

            if car_lines[jj][0][2]  > x_least and car_lines[jj][0][2]  < x_high and \
            car_lines[jj][0][3] > y_least and car_lines[jj][0][3] < y_high:
                flag = 1 

        if flag == 1:
            cv2.fillPoly(img, [pts], (0,0,255))
            #cv2.polylines(img, [pts], True, (0,0,255), 3)
        else:
            cv2.fillPoly(img, [pts],(0,255,0))


if __name__ == "__main__":

    param=[1,0.001,10,60,40]
    filen="example.jpg"

    """
    rho=rho,theta=angle,threshold=thresh,lines=np.array([]),minLineLength=min_length,maxLineGap=max_gap)
    """
    if not os.path.isfile("output.npy"):
        city_segment.segmentation(filen, "output.jpg")

    segments = np.load("output.npy")

    mask_road = city_segment.largest_connected_component()
    img = cv2.imread(filen)

    # Apply traditional CV to masked image
    lines = get_lines(img,  mask_road.astype(np.uint8), param)
    h_lines = lines_processing(lines)
    get_scatter(h_lines)
    #print(np.shape(h_lines))
    X,y_hc,n_clust = hc()
    clust_lines = parking_spots(X,y_hc,n_clust)
    clust_lines = lines_processing(clust_lines)
    print(clust_lines)
    line_image = draw_lines(img, clust_lines)
    cv2.imwrite("woslope.png",line_image)
    
   # imsave("wslope.png",line_image_slope)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Save images
    cv2.imwrite("mask_image.png",gray_image*mask_road)