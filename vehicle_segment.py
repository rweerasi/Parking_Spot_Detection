from scipy.misc import imsave
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data, utils


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

    # 
    width, height = orig_img.shape[1], orig_img.shape[0]
    masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    # identical to Faster RCNN object detection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
    plt.show()

if __name__ == "__main__":
    segmentation("example.jpg")
