import cv2
from PIL import Image
import numpy as np
import os.path
import matplotlib.pyplot as plt
from scipy.misc import imsave

# Our libraries
import city_segment
import parking_detection
import vehicle_segment


if __name__ == "__main__":
    
    filename = "example.jpg"

    # High level segmentation using pre-trained network
    if not os.path.isfile("output.npy"):
        city_segment.segmentation(filename, "output.png")

    # Create mask for only largest road segment
    mask = city_segment.largest_connected_component(filename="output.npy")

    # Apply traditional CV to masked image
    img = cv2.imread(filename)
    lines = parking_detection.get_lines(img, mask, [1, 0.001, 10, 60, 40])
    #line_image = parking_detection.draw_lines(img, lines)
        
    # Post-processing on lines 
    h_lines = parking_detection.lines_processing(lines)
    parking_detection.get_scatter(h_lines)
    x, y, n = parking_detection.hc()
    clust_lines = parking_detection.parking_spots(x, y, n)
    clust_lines = parking_detection.lines_processing(clust_lines)
    clust_lines = parking_detection.extend_lines(clust_lines)
    #line_image = parking_detection.draw_lines(img, clust_lines)
    parking_detection.highlight_spots(img, clust_lines)
    
    cv2.imshow("", img)
    cv2.waitKey(-1)


    # Detect vehicles in image
    #car_lines = vehicle_segment.segmentation(filename)
        
    # 


    # Save images
    imsave("line_image.png", line_image)
    plt.figure()
    plt.imshow(line_image)
    plt.show()
