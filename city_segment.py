import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
from deeplabv3 import DeepLabV3
from scipy.misc import imsave
import numpy as np
from PIL import Image


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
    model.load_state_dict(torch.load("model_13_2_2_2_epoch_580.pth"))
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
