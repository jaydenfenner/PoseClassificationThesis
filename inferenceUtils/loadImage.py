import numpy as np
import cv2
from skimage import io
import torchvision.transforms as transforms
import torch
from inferenceUtils.constants import constants, CoverType, getModelProperties, PretrainedModels

def cropAndRotate_D455(npImage, runName: str) -> np.ndarray:
    '''
    Crop and rotate D455 depth numpy array for further preprocessing
    - Rotate 90 deg to match SLp direction
    - Crop to square containing bed
    - runName options:
        - D455_V1 (double bed, spare room)
        - D455_V3 (single mattress, cinema)
    '''
    cropped = None
    if (runName == 'D455_V1'):
        cropped = crop_D455_custom(npImage, heightScale=0.99, pShiftDown=-0.02, pShiftRight=0.0)
    if (runName == 'D455_V2'):
        cropped = crop_D455_custom(npImage, heightScale=0.85, pShiftDown=0.03, pShiftRight=0.01)
    elif (runName == 'D455_V3'):
        cropped = crop_D455_custom(npImage, heightScale=0.85, pShiftDown=0.03, pShiftRight=0.01)
    elif (runName == 'D455_S01'):
        cropped = crop_D455_custom(npImage, heightScale=0.85, pShiftDown=0.03, pShiftRight=0.02)
    else:
        raise ValueError(f'runName not recognised: [{runName}]')
    return cropped

def crop_D455_custom(npImage, heightScale: float, pShiftDown: float, pShiftRight: float) -> np.ndarray:
    '''
    Rotate 90deg then crop D455 depth numpy array for further preprocessing
    - Scale by a factor of original height (after rotating)
    - shift down and right by % of original (after rotating)
        - provide values in range [-1, 1]
    '''
    rotated_image = np.rot90(npImage) # rotate 90 degrees anticlockwise

    # crop to square and shift
    origHeight, origWidth = rotated_image.shape[:2]
    newSize = int(origWidth * heightScale)
    newMinY = (origHeight - newSize) // 2 + int(origHeight * pShiftDown)
    newMinX = (origWidth - newSize) // 2 + int(origWidth * pShiftRight)
    cropped = rotated_image[newMinY:newMinY+newSize, newMinX:newMinX+newSize]

    return cropped

D455_DEFAULT_WIDTH = 640
D455_DEFAULT_HEIGHT = 360
depth_type = np.uint16 # metatada csv states 2 bytes per pixel
def readD455DepthRaw(path: str, width = D455_DEFAULT_WIDTH, height = D455_DEFAULT_HEIGHT) -> np.ndarray:
    '''
    Read D455 depth image and convert to numpy array
    - Output is np.uint16 
    '''
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
    # construct path, read depth image png and convert to numpy array
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
    # resize image to required input size
    model_input_size = getModelProperties(modelType).model_input_size # = [256, 256]
    newSize = model_input_size[0]
    img_input = cv2.resize(npImage, (newSize, newSize), interpolation=cv2.INTER_AREA)

    # convert to datatype supported by pytorch (float in format h,w,c)
    img_input = img_input.astype(np.float32)
    img_input = img_input[..., None]

    #! DISPLAY NORMALISATION INFORMATION FOR DEBUGGING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print(f"\n---- NORMALISING ------")
    print(f"norm mean: {constants.mean_depth}")
    print(f"norm std: {constants.std_depth}")
    print(f"pre-norm -->")
    print(f"	img_mean: {img_input.mean()}")
    print(f"	img_std: {img_input.std()}")
    print(f"	img_max: {img_input.max()}")
    print(f"	img_min: {img_input.min()}")
    #! DISPLAY NORMALISATION INFORMATION FOR DEBUGGING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # 0-255
    # img / 255

    # convert to tensor and normalise
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.mean_depth], std=[constants.std_depth]) # mean = 0.7302197, std = 0.25182092
        ])
    img_input = torch_transforms(img_input)

    #! DISPLAY NORMALISATION INFORMATION FOR DEBUGGING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print(f"post-norm -->")
    print(f"	img_mean: {img_input.mean()}")
    print(f"	img_std: {img_input.std()}")
    print(f"	img_max: {img_input.max()}")
    print(f"	img_min: {img_input.min()}")
    #! DISPLAY NORMALISATION INFORMATION FOR DEBUGGING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # displayTorchImg(img_input) #? display image for debugging purposes
    # print(img_input.shape) #? display image shape for debugging purposes

    return img_input