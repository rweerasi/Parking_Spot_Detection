from scipy.misc import imsave
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data, utils


def cars_only(array, ids, scores, threshold):
    #a = array
    # return (np.array(a)[np.where(np.array(ids) == 2.), :])[0].tolist()
    output = []
    for ii, id in enumerate(ids):
        # Check if the label is a car
        if id[0] == 2. and scores[ii] > threshold:
            output.append(array[ii])
    return output



def segmentation(filename, output_file="output.png"):
    """
    Input:
        filename:   location of (1024x2048x3) png input image
        output:     filename for segmented output image, size (1024x2048x1) with 
                    labels 0-19 for each pixel 
    Outputs:
        output file saved as given name
    """
    net = model_zoo.get_model("mask_rcnn_resnet50_v1b_coco", pretrained=True)
    x, orig_img = data.transforms.presets.rcnn.load_test(filename)


    # Run network and store results
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

    # Get only the cars
    ids_car = cars_only(ids, ids, scores, 0.6)
    scores_car = cars_only(scores, ids, scores, 0.6)
    bboxes_car = cars_only(bboxes, ids, scores, 0.6 )
    masks_car = cars_only(masks, ids, scores, 0.6)

    # Formatting
    width, height = orig_img.shape[1], orig_img.shape[0]
    """
    masks = utils.viz.expand_mask(masks_car, bboxes_car, (width, height), scores_car)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    # identical to Faster RCNN object detection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
    plt.show()
    """

    # Draw bottom bounding boxes on image
    lines = []
    for bbox in bboxes_car:
        x_1 = bbox[2]
        x_2 = bbox[0]
        y = max(bbox[1], bbox[3])
        cv2.line(orig_img, (x_1, y), (x_2, y), (255, 0, 0), 5)
        lines.append([[int(2*x_1), int(2*y), int(2*x_2), int(2*y)]])
    cv2.imshow("", orig_img)
    cv2.waitKey(-1)

    return lines


if __name__ == "__main__":
    segmentation("example.jpg")
