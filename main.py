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

    # Create mask for only road 
    segments = np.load("output.npy")
    mask = 255 * (segments == 0).astype(np.uint8)

    # TODO: Should probably put this in its own function
    # Get largest connected component
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - mask)
    sizes = stats[:, -1]
    max_label = np.argmax(sizes)
    mask_road = np.zeros(mask.shape)
    mask_road[labels == max_label] = 255
    print(labels.shape)

    # Apply mask 
    mask = np.dstack((mask_road, mask_road, mask_road)).astype(np.uint8)
    img = cv2.imread("example.png")
    masked_image = parking_detection.mask_image(img, mask)

    # Apply traditional CV to masked image
    lines = parking_detection.get_lines(img, 255 - mask_road.astype(np.uint8))
    line_image = parking_detection.draw_lines(img, lines)

    imsave("line_image.png", line_image)
    plt.figure()
    plt.imshow(line_image)
    plt.show()
