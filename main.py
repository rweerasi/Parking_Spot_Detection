import cv2
from PIL import Image
import numpy as np
import os.path
import matplotlib.pyplot as plt
from scipy.misc import imsave

# Our libraries
import city_segment
import parking_detection


if __name__ == "__main__":
    
    # High level segmentation using pre-trained network
    if not os.path.isfile("output.npy"):
        city_segment.segmentation("example.png", "output.png")

    # Create mask for only largest road segment
    mask = city_segment.largest_connected_component(filename="output.npy")

    # Apply traditional CV to masked image
    img = cv2.imread("example.png")
    lines = parking_detection.get_lines(img, mask.astype(np.uint8))
    line_image = parking_detection.draw_lines(img, lines)

    # Save images
    imsave("line_image.png", line_image)
    plt.figure()
    plt.imshow(line_image)
    plt.show()
