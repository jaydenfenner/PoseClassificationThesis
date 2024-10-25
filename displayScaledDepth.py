'''
Super simple script purely to generate a properly scaled, 
colour-mapped depth image for argument about information
present in depth images compared with RGB
'''

from inferenceUtils.croppingSimLab import cropDepthPngFromSimLab, cropNpImage_scaleY_XY
from skimage import io
import numpy as np
from inferenceUtils.imageUtils import display_depth_image_with_colormap
import cv2
from D455_testing.comparePreprocessing import display_depth_histogram

def main():
    relativePath = 'SLP/simLab/00004/depth/uncover/image_000001.png'
    subject = 4
    subjPeak = 187
    scaleFactor = 11.0

    img = io.imread(relativePath) # read the SLP image
    npImg =  np.array(img) # convert to numpy array
    cropped, newMinY, newMinX = cropDepthPngFromSimLab(npImg, subj=subject)
    croppedAndScaled = scale_and_shift_255(cropped, scaleFactor, subjPeak)
    
    display_depth_image_with_colormap(croppedAndScaled, 'SLP-cropped', persistImages=True)
    display_depth_histogram(croppedAndScaled, 'SLP-cropped', bins=200)
    # cv2.imwrite(f's04_p01_u_[{scaleFactor}_{subjPeak}].png', cv2.applyColorMap(croppedAndScaled, cv2.COLORMAP_JET))

    # Wait for any key press to close the window
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    
def scale_and_shift_255(npImage: np.ndarray, scaleFactor: float, subjPeak: int):
    '''scale uint8 numpy image by scaleFactor and centre subject in scale'''
    image = npImage.astype(np.int) #! temp convert to signed int

    scaled_img = (image - subjPeak) * scaleFactor + 128
    
    # Step 2: Clip the result to ensure values are in the valid range [0, 255]
    scaled_img = np.clip(scaled_img, 0, 255)
    scaled_img = scaled_img.astype(np.uint8) #! convert back to original
    
    return scaled_img

if __name__ == '__main__':
    main()