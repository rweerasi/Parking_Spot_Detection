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

def line_dist(line1, line2):

    return np.linalg.norm(line1-line2)

'''
def get_slopes(img,lines):

    lines = np.array(lines)
    x_vec = lines[:,:,0] - lines[:,:,2]
    y_vec = lines[:,:,1] - lines[:,:,3]
    print(lines)
    slope = np.divide(y_vec,x_vec)
    intercept = lines[:,:,1] - np.multiply(lines[:,:,0],slope)
    n_lines, _ = np.shape(intercept)

    new_lines = np.zeros((n_lines*n_lines,1,4))
    k = 0

    thresh = 100

    for i in range(0,n_lines):
        for j in range(i, n_lines):

            if (abs(slope[i,0] - slope[j,0]) < 0.1 and abs(intercept[i,0] - intercept[j,0]) < 2):
                if (lines[i,0,0] > lines[j,0,0]):

                    dist = line_dist(lines[j,0,2:4],lines[i,0,0:2])

                    if dist < thresh:
                        new_lines[k,0,0] = lines[j,0,0]
                        new_lines[k,0,1] = lines[j,0,1]
                        new_lines[k,0,2] = lines[i,0,2]
                        new_lines[k,0,3] = lines[i,0,3]

                else:

                    dist = line_dist(lines[i,0,2:4],lines[j,0,0:2])

                    if dist < thresh:

                        new_lines[k,0,0] = lines[i,0,0]
                        new_lines[k,0,1] = lines[i,0,1]
                        new_lines[k,0,2] = lines[j,0,2]
                        new_lines[k,0,3] = lines[j,0,3]

            k = k+1

    new_lines = new_lines[:k,:,:]
    new_lines = new_lines.astype(int)

    return new_lines
'''
def get_scatter(lines):

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

    #y_points = np.flip(x_points)

    X = np.vstack((x_points,y_points))

    plt.scatter(x_points,y_points)
    np.savetxt("foo.csv", X, delimiter=",")

    plt.show()

def hc():

    X = np.genfromtxt('foo.csv',delimiter=',')

    X = np.transpose(X)

    X[:,1] = 10*X[:,1]

    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.ylabel('Skewed Euclidean distances')
    plt.show()

    #print(sch.linkage(X, method = 'ward')[-30:,2])


    n_clust = 9
    # Fitting Hierarchical Clustering to the dataset
    hc = AgglomerativeClustering(n_clusters = n_clust, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)

    X[:,1] = np.round(0.1*X[:,1])

    # Visualising the clusters
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 100, c = 'yellow', label = 'Cluster 6')
    plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 100, c = 'brown', label = 'Cluster 7')
    plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 100, c = 'black', label = 'Cluster 8')
    plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s = 100, c = 'orange', label = 'Cluster 9')
    #plt.scatter(X[y_hc == 9, 0], X[y_hc == 9, 1], s = 100, c = 'purple', label = 'Cluster 10')
    #plt.scatter(X[y_hc == 10, 0], X[y_hc == 10, 1], s = 100, c = 'pink', label = 'Cluster 11')
    
    plt.title('Clusters of parking lines')
    
    return X, y_hc, n_clust

def parking_spots(X,y_hc,n_clust):


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
        


def get_harris_corners(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[255,0,0]
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return dst

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


def extend_lines(lines):

    # Separate into left and right 
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

def highlight_spots(img, long_lines):

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
    
    # Draw rectangles 
    for ii in range(len(left_lines) - 1):
        a = np.array(ordered_lines[ii]).reshape((2,2))
        b = np.flipud(np.array(ordered_lines[ii+1]).reshape((2,2)))
        pts = np.vstack((a, b))
        cv2.polylines(img, [pts], True, (0, 0, 255), 3)

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
        cv2.polylines(img, [pts], True, (0,0,255), 3)


        

if __name__ == "__main__":

    #img=cv2.imread("parking_2.png")
    #height, width, channels = img.shape
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
    #new_lines = get_slopes(img,h_lines)
    #new_lines = lines_processing(new_lines)
    line_image = draw_lines(img, clust_lines)
    #line_image_slope = draw_lines(img, new_lines)
    cv2.imwrite("woslope.png",line_image)
    
   # imsave("wslope.png",line_image_slope)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Save images
    cv2.imwrite("mask_image.png",gray_image*mask_road)

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
