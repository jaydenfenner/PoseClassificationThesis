import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from inferenceUtils.croppingSimLab import cropDepthPngFromSimLab
from inferenceUtils.loadImage import prepareNpDepthImgForInference
from inferenceUtils.imageUtils import display_depth_image_with_colormap, displayTorchImg, torchInputToNumpy

''' 
TLDR: run this script from the repo root as: 
    python -m D455_testing.comparePreprocessing

NOTE:
    To run with imports working correctly, this script must be run as a module 
    from the root directory. This allows relative imports to all start from the 
    root of the repo. This also enables relative imports when functions are used
    in other scripts
'''

#! REMEMBER TO IMPORT FUNCTION FOR PREPROCESSING + SCALING SQUARE NUMPY ARRAY

def main():
    #************ For SLP:
    SLP_cropped = readandCropSLPDepthImg('D455_testing/SLP_subj01_u_p014.png') # read and crop to square for preprocessing
    # save_array_to_csv(SLP_image)
    # display_depth_histogram(SLP_image, 'SLP-cropped')
    display_depth_image_with_colormap(SLP_cropped, 'SLP-cropped', persistImages=True)

    SLP_tensor_input = prepareNpDepthImgForInference(SLP_cropped)
    displayTorchImg(SLP_tensor_input, 'SLP-input', persistImages=True)
    display_depth_histogram(torchInputToNumpy(SLP_tensor_input), 'SLP--tensor-input')

    #************** for D455:
    D455_image = readD455DepthRaw('D455_testing/D455_S01_u.raw') # read image - pre-crop
    # # save_array_to_csv(D455_image)
    # display_depth_histogram(D455_image, 'D455-original')
    # display_depth_image_with_colormap(D455_image,'D455-original', persistImages=True)

    D455_cropped = cropAndRotate_D455(D455_image)
    # display_depth_histogram(D455_image, 'D455-cropped+rotated')
    display_depth_image_with_colormap(D455_cropped,'D455-cropped', persistImages=True)

    D455_tensor_input = prepareNpDepthImgForInference(D455_cropped)
    displayTorchImg(D455_tensor_input, 'D455-input', persistImages=True)
    display_depth_histogram(torchInputToNumpy(D455_tensor_input), 'D455--tensor-input')

    # Wait for any key press to close the window
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return

def readandCropSLPDepthImg(relativePath: str):
    img = io.imread(relativePath) # read the SLP image
    npImg =  np.array(img) # convert to numpy array
    cropped, newMinY, newMinX = cropDepthPngFromSimLab(npImg, subj=1) # crop image (hardcode subj 1 cropping)
    return cropped

def prepD455DepthRaw(path: str):
    '''Preprocess D455 raw depth image into tensor for inference'''
    original = readD455DepthRaw('path goes here')
    cropped = cropAndRotate_D455(original)
    return cropped



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




def save_array_to_csv(array, filename="depth_image.csv"):
    '''Helper function to save numpy array to .csv for inspection'''
    if array.ndim != 2: # Ensure the input is a 2D NumPy array
        raise ValueError("Input array must be 2D.")
    # Save the array to a CSV file
    np.savetxt('D455_testing/'+filename, array, delimiter=",", fmt="%d")
    print(f"Array saved as root/D455_testing/{filename}")


def display_depth_histogram(depth_image, name: str):
    '''plot histagram of depth image'''
    if depth_image.ndim != 2: # Ensure the input is a 2D NumPy array
        raise ValueError("Input depth image must be a 2D array.")
    plt.hist(depth_image.ravel(), bins=10, edgecolor='black') # Plot the histogram with 10 bins
    # Add labels and title for clarity
    plt.xlabel('Pixel Value'); plt.ylabel('Frequency')
    plt.title('Depth Image Histogram (10 Bins) for img: '+ name)
    plt.show() # Show the plot



if __name__ == '__main__':
    main()