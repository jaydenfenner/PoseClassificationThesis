import numpy as np
import cv2
from skimage import io
import torchvision.transforms as transforms
from inferenceUtils.constants import constants, CoverType, getModelProperties, PretrainedModels

def cropAndRotate_D455(npImage) -> np.ndarray:
    '''
    Crop and rotate D455 depth numpy array for further preprocessing
    - Rotate 90 deg to match SLp direction
    - Crop to square containing bed
    '''
    rotated_image = np.rot90(npImage) # rotate 90 degrees anticlockwise

    # crop to square and shift
    shiftPercentageYX = [-0.02, 0] # down and right
    origHeight, origWidth = rotated_image.shape[:2]
    newSize = int(origWidth * 0.99)
    newMinY = (origHeight - newSize) // 2 + int(origHeight * shiftPercentageYX[0])
    newMinX = (origWidth - newSize) // 2 + int(origWidth * shiftPercentageYX[1])
    cropped = rotated_image[newMinY:newMinY+newSize, newMinX:newMinX+newSize]

    return cropped

D455_DEFAULT_WIDTH = 640
D455_DEFAULT_HEIGHT = 360
depth_type = np.uint16 # metatada csv states 2 bytes per pixel
def readD455DepthRaw(path: str, width = D455_DEFAULT_WIDTH, height = D455_DEFAULT_HEIGHT) -> np.ndarray:
    '''Read D455 depth image and convert to numpy array'''
    # Read the raw depth data
    with open(path, 'rb') as f:
        depth_image = np.frombuffer(f.read(), dtype=depth_type)
    # Reshape to the correct dimensions (width, height)
    depth_image = depth_image.reshape((height, width))
    return depth_image


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

def prepareNpDepthImgForInference(npImage, modelType: PretrainedModels):
    '''
    take square numpy image\n
    resize, normalise and convert to tensor for inference with pytorch
    - output is torch.Size([1, 256, 256])
    '''
    #! resize image to required input size
    model_input_size = getModelProperties(modelType).model_input_size
    newSize = model_input_size[0]
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
    # print(img_input.shape) #! display image shape for debugging purposes
    return img_input