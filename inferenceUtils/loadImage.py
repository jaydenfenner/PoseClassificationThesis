import numpy as np
import cv2
from skimage import io
import torchvision.transforms as transforms
from inferenceUtils.constants import constants, CoverType

def readDepthPngFromSimLab(subj=1, cover: CoverType = CoverType.UNCOVER, poseNum=1):
    '''
    Take an SLP simLab subject, cover and pose number and return the image from simLab\n
    NOTE: subj from 1-7\n
    NOTE: cover from constants.CoverType
    NOTE: poseNum from 1-45\n
    RETURNS:\n\n2D numpy array read from greyscale depth image
    '''
    #! construct path, read depth image png and convert to numpy array
    subjWithZeros = "{:05d}".format(subj)
    poseNumWithZeros = "{:06d}".format(poseNum)
    path_to_depthImg = f'SLP/simLab/{subjWithZeros}/depth/{cover.value}/image_{poseNumWithZeros}.png'
    img = io.imread(path_to_depthImg)
    img = np.array(img)
    return img

def prepareNpDepthImgForInference(npImage):
    '''
    take square numpy image\n
    resize, normalise and convert to tensor for inference with pytorch
    - output is torch.Size([1, 256, 256])
    '''
    #! resize image to required input size
    newSize = constants.model_input_size[0]
    img_input = cv2.resize(npImage, (newSize, newSize), interpolation=cv2.INTER_AREA)

    #! convert to datatype supported by pytorch (float in format h,w,c)
    img_input = img_input.astype(np.float32)
    img_input = img_input[..., None]

    #! convert to tensor and normalise
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.mean_depth], std=[constants.std_depth])
        ])
    img_input = torch_transforms(img_input) #! imagePatch_Pytorch

    # displayTorchImg(img_input) #! display image for debugging purposes
    print(img_input.shape)
    return img_input