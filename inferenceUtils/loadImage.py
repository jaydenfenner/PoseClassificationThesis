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
    '''
    #! construct path, read depth image png and convert to numpy array
    subjWithZeros = "{:05d}".format(subj)
    poseNumWithZeros = "{:06d}".format(poseNum)
    path_to_depthImg = f'SLP/simLab/{subjWithZeros}/depth/{cover.value}/image_{poseNumWithZeros}.png'
    img = io.imread(path_to_depthImg)
    img = np.array(img)
    return img

def preparePngForInference(img):
    #! resize image to required input size
    newSize = constants.model_input_size[0]
    img_input = cv2.resize(img, (newSize, newSize), interpolation=cv2.INTER_AREA)

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
    return img_input