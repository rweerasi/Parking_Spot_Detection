import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
from deeplabv3 import DeepLabV3
from scipy.misc import imsave
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def segmentation(filename, output_file="output.png"):
    """
    Input:
        filename:   location of (1024x2048x3) png input image
        output:     filename for segmented output image, size (1024x2048x1) with 
                    labels 0-19 for each pixel 
    Outputs:
        output file saved as given name
    """
    # Load pre-trained network
    model = DeepLabV3(1, ".")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("model_13_2_2_2_epoch_580.pth"))
    else:
        model.load_state_dict(torch.load("model_13_2_2_2_epoch_580.pth", map_location=lambda storage, loc: storage))
    
    model.eval()

    # Preprocess input
    transformation = transforms.Compose([
            transforms.ToTensor(),
    ])
    #image = Image.open("stuttgart_00_000000_000080_leftImg8bit.png")
    image = Image.open(filename)
    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Run model
    input = Variable(image_tensor)
    output = model(input)

    # Output
    out_img = np.argmax(output.detach().numpy(), axis=1).astype(np.uint8)[0,:,:]
    np.save("output.npy", out_img)
    imsave(output_file, out_img)


def largest_connected_component(filename="output.npy"):
    """
    Input:
        filename:   file name for segmented image, saved as npy or npz
    Output:
        output:     numpy array with 255 in largest connected component and 0 for all
                    other regions
    """
    img = np.load("output.npy")
    img = (img == 0).astype(np.uint8) 
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
    sizes = stats[1:, -1]
    max_label = np.argmax(sizes) 
    print(sizes)
    return 255 * (output == max_label + 1).astype(np.uint8)
    
def closing(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def dilation(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

if __name__ == "__main__":
    # segmentation("example.png")
    img = largest_connected_component()
    img = dilation(img, 30)
    plt.imshow(img)
    plt.colorbar()
    plt.show()
