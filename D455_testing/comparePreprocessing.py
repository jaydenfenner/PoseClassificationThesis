import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage import io
from inferenceUtils.croppingSimLab import cropDepthPngFromSimLab, cropNpImage_scaleY_XY
from inferenceUtils.loadImage import prepareNpDepthImgForInference
from inferenceUtils.imageUtils import display_depth_image_with_colormap, displayTorchImg, torchInputToNumpy, scaleNpImageTo255
from inferenceUtils.loadImage import readD455DepthRaw, cropAndRotate_D455
from inferenceUtils.constants import PretrainedModels

''' 
TLDR: run this script from the repo root as: 
    python -m D455_testing.comparePreprocessing

NOTE:
    To run with imports working correctly, this script must be run as a module 
    from the root directory. This allows relative imports to all start from the 
    root of the repo. This also enables relative imports when functions are used
    in other scripts
'''

def main():
    modelType = PretrainedModels.HRPOSE_DEPTH
    #************ For SLP:
    if (False):
        # read and crop to square for preprocessing
        if(False): # true for simlab
            SLP_cropped = readandCropSLPDepthImg('D455_testing/simLab_subj01_u_p014.png') #? use for simlab sample
        else:
            SLP_cropped = readandCropSLPDepthImg('D455_testing/danaLab_sample_u.png', cropParams_scaleY_x_y=[0.7, 0.12, 0.])  #? use for danaLab sample

        # save_array_to_csv(SLP_image)
        # display_depth_histogram(SLP_image, 'SLP-cropped')
        display_depth_image_with_colormap(SLP_cropped, 'SLP-cropped', persistImages=True)

        SLP_tensor_input = prepareNpDepthImgForInference(SLP_cropped, modelType)
        # plot_tensor_cumulative_distribution(SLP_tensor_input, 'SLP_tensor_input')

        displayTorchImg(SLP_tensor_input, 'SLP-input', persistImages=True)

        if (False): # show np histogram and cumulative distribution
            SLP_tensor_np = torchInputToNumpy(SLP_tensor_input)
            display_depth_histogram(SLP_tensor_np, 'SLP--tensor-input')
            display_cumulative_distribution(SLP_tensor_np, 'SLP_tensor_np')

    # Read preprocessed D455 as png to test difference (NOTE IT MADE NO DIFF)
    if (False):
        savedPNG_cropped = readandCropSLPDepthImg('D455_testing/savedDepthImg.png', cropParams_scaleY_x_y=[1.0, 0., 0.])  #? no crop or scaling
        display_depth_image_with_colormap(savedPNG_cropped, 'savedPNG_cropped', persistImages=True)
        savedPNG_tensor_input = prepareNpDepthImgForInference(savedPNG_cropped, modelType)
        displayTorchImg(savedPNG_tensor_input, 'savedPNG_tensor_input', persistImages=True)
    
    
    #************** for D455 (initial run):
    if (False):
        # for thresh in [2100, 2150, 2200, 2250, 2300]: #? 2200 was best
        thresh = 2200

        D455_image = readD455DepthRaw('D455_testing/D455_S01_u.raw') # read image - pre-crop
        # # save_array_to_csv(D455_image)
        # display_depth_histogram(D455_image, 'D455-original')
        # display_depth_image_with_colormap(D455_image,'D455-original', persistImages=True)

        D455_cropped = cropAndRotate_D455(D455_image, runName='D455_V1') #? note run name must match option in cropAndRotate_D455
        # display_depth_histogram(D455_image, 'D455-cropped+rotated')
        # display_depth_image_with_colormap(D455_cropped,'D455-cropped', persistImages=True)

        #! D455_masked = mask_D455(D455_cropped)
        # print(f"max in cropped: {D455_cropped.max()}")
        # print(f"avg in cropped: {D455_cropped.mean()}")
        D455_cropped = threshold_depth_image(D455_cropped, threshold_value=thresh)
        D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0.1, 0.16], margin_percent_y=[0.03, 0.])

        D455_tensor_input = prepareNpDepthImgForInference(D455_cropped, modelType)
        displayTorchImg(D455_tensor_input, f'D455-input{thresh}', persistImages=True)

        if (False): # display histograms
            D455_tensor_np = torchInputToNumpy(D455_tensor_input)
            # display_depth_histogram(D455_tensor_np, 'D455--tensor-input')
            display_cumulative_distribution(D455_tensor_np, 'D455_tensor_np')

    #************** for D455 (third run):
    if (False):
        # for thresh in [2650, 2690, 2700, 2710]: #? 2690 was best
        thresh = 2690
        
        D455_image = readD455DepthRaw('D455_cinema2/200_Depth.raw') # read image - pre-crop
        # # save_array_to_csv(D455_image)
        # display_depth_histogram(D455_image, 'D455-original')
        # display_depth_image_with_colormap(D455_image,'D455-original', persistImages=True)

        D455_cropped = cropAndRotate_D455(D455_image, runName='D455_V3') #? note run name must match option in cropAndRotate_D455
        # display_depth_histogram(D455_image, 'D455-cropped+rotated')
        # display_depth_image_with_colormap(D455_cropped,'D455-cropped', persistImages=True)

        #! D455_masked = mask_D455(D455_cropped)
        print(f"max in cropped: {D455_cropped.max()}")
        print(f"max in cropped: {D455_cropped.min()}")
        print(f"avg in cropped: {D455_cropped.mean()}")
        D455_cropped = threshold_depth_image(D455_cropped, threshold_value=thresh)
        D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0.2, 0.25], margin_percent_y=[0.03, 0.])

        if (False): D455_cropped, min, max = truncate_min_to_percent(D455_cropped, percentile=0.983) # truncate close pixels (artifacts) to achieve correct scale
        D455_cropped = truncate_min_to_threshold(D455_cropped, threshold=1250) # truncate close pixels (artifacts) to achieve correct scale

        D455_tensor_input = prepareNpDepthImgForInference(D455_cropped, modelType)
        displayTorchImg(D455_tensor_input, f'D455-input{thresh}', persistImages=True)

        if (True): # display histograms
            D455_tensor_np = torchInputToNumpy(D455_tensor_input)
            display_depth_histogram(D455_tensor_np, 'D455--tensor-input', bins=200)
            # display_cumulative_distribution(D455_tensor_np, 'D455_tensor_np')

    
    #************** for D455 (third run SECOND ATTEMPT AT PREPROCESSING):
    if (False):
        # for thresh in [2650, 2690, 2700, 2710]: #? 2690 was best
        thresh = 2690
        
        D455_image = readD455DepthRaw('D455_cinema2/200_Depth.raw') # read image - pre-crop
        # # save_array_to_csv(D455_image)
        # display_depth_histogram(D455_image, 'D455-original')
        # display_depth_image_with_colormap(D455_image,'D455-original', persistImages=True)


        D455_cropped = cropAndRotate_D455(D455_image, runName='D455_V3') #? note run name must match option in cropAndRotate_D455
        # display_depth_histogram(D455_image, 'D455-cropped+rotated')
        # display_depth_image_with_colormap(D455_cropped,'D455-cropped', persistImages=True)


        #TODO convert image to 0-255 for ease of use
        D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=False)
        D455_cropped = scaleNpImageTo255(D455_cropped) # again for sanity check

        # TODO remove ladder by cropping margin to max
        D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0., 0.25], margin_percent_y=[0., 0.])

        # display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) #! histogram pre-scaling (use to find current bed min and max)
        # TODO scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
        D455_cropped = scale_to_fit_range(D455_cropped, 
                                          currentMin=204, currentMax=232, # current based on histogram before scaling
                                          desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
        # display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) #! histogram post-scaling (bed min/max now match danaLab)

        # TODO threshold background to match danaLab (set background to peak=220)
        D455_cropped = threshold_depth_image(D455_cropped, threshold_value=196, background_value=220)

        # TODO set top left pixel to force correct histogram scale
        D455_cropped[0][0] = 255

        # save_depth_as_greyscale_png(D455_cropped, 'D455_testing/savedDepthImg.png') #! temp save as png

        # display_depth_image_with_colormap(D455_cropped,'D455_crop_scale_shift_mask', persistImages=True)
        # display_depth_histogram(D455_cropped, 'D455_crop_scale_shift_mask', bins=50) #! histogram post-thresholding (floor now match danaLab)

        D455_tensor_input = prepareNpDepthImgForInference(D455_cropped, modelType)
        displayTorchImg(D455_tensor_input, f'D455-input{thresh}', persistImages=True)

        if (False): # display histograms
            display_depth_histogram(D455_cropped, 'D455_cropped', bins=200)
            # display_cumulative_distribution(D455_tensor_np, 'D455_tensor_np')

    #************** for D455 (initial run SECOND ATTEMPT AT PREPROCESSING --> replicate V3 above):
    if (False):
        if (False):
            D455_image = readD455DepthRaw('D455_testing/D455_S01_u.raw') # read image - pre-crop
        elif(False):
            D455_image = readD455DepthRaw('D455_testing/D455_S04_c.raw') # read image - pre-crop
        else:
            D455_image = readD455DepthRaw('D455_testing/D455_S07_u.raw') # read image - pre-crop

        D455_cropped = cropAndRotate_D455(D455_image, runName='D455_V1') #? note run name must match option in cropAndRotate_D455
        D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=True)
        D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0.1, 0.155], margin_percent_y=[0.041, 0.])

        # display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) # TODO bed (min, max) = (180, 207)
        #? scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
        D455_cropped = scale_to_fit_range(D455_cropped, 
                                          currentMin=180, currentMax=200, # current based on histogram before scaling
                                          desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
        # display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) #! histogram post-scaling (bed min/max now match danaLab)

        D455_cropped = threshold_depth_image(D455_cropped, threshold_value=189, background_value=220)

        display_depth_image_with_colormap(D455_cropped,'D455-cropped', persistImages=True)

    #************** for D455 (V2 --> replicate V3 above):
    if (True):
        valid_image_numbers = [1, 2, 3, 6, 7] # note 6, 7 are WITH COVER versions of 1, 3
        imageNumber = valid_image_numbers[3]
        D455_image = readD455DepthRaw(f'D455_cinema1/{imageNumber}_Depth.raw') # read image - pre-crop

        D455_cropped = cropAndRotate_D455(D455_image, runName='D455_V2') #? note run name must match option in cropAndRotate_D455
        D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=True)
        # D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0., 0.2], margin_percent_y=[0., 0.]) #? margin not required
        D455_cropped = apply_mask_with_line(D455_cropped, point=(0.779, 0.5), angle_degrees=88, maskBelow=True)
        D455_cropped = apply_circular_mask(D455_cropped, center=(1.2, 0.675), radius_ratio=0.45)

        # display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) # TODO bed (min, max) = (180, 207)
        #? scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
        D455_cropped = scale_to_fit_range(D455_cropped, 
                                          currentMin=201, currentMax=230, # current based on histogram before scaling
                                          desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
        # display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) #! histogram post-scaling (bed min/max now match danaLab)

        D455_cropped = threshold_depth_image(D455_cropped, threshold_value=185, background_value=220)

        display_depth_image_with_colormap(D455_cropped,'D455-cropped', persistImages=True)
        display_depth_histogram(D455_cropped, 'D455_cropped', bins=200) #! histogram post-scaling (bed min/max now match danaLab)


    #************** compare histograms of cropped SLP:
    '''
    - simLab cropped(0-255):
        - bed:  min: 180,  peak 190,  max: 208
        - floor:  min: 237,  peak 242,  max: 246
        - floor-bed peak diff = 52  (21.5% of floor peak)  (27% of bed peak)
        - body-bed range = 28
    - danaLab cropped (0-255):
        - bed:  min: 162,  peak 179,  max: 186 --------> min_on_side: 152
        - floor:  min: 212,  peak 220,  max: 227
        - floor-bed peak diff = 41  (18.6% of floor peak)  (23% of bed peak)
        - body-bed range = 24
    - D455_V3 cropped (0-255):
        - bed:  min: 204,  peak 222,  max: 232
    '''
    
    if(False):
        # read and crop to square for preprocessing
        SLP_simLab_cropped = readandCropSLPDepthImg('D455_testing/simLab_subj01_u_p014.png') #? use for simlab sample
        if (False):
            SLP_danaLab_cropped = readandCropSLPDepthImg('D455_testing/danaLab_sample_u.png', cropParams_scaleY_x_y=[0.7, 0.12, 0.])  #? use for danaLab sample
        else:
            SLP_danaLab_cropped = readandCropSLPDepthImg('D455_testing/danaLab_side_sample_u.png', cropParams_scaleY_x_y=[0.7, 0.12, 0.])  #? use for danaLab sample

        # display images
        display_depth_image_with_colormap(SLP_simLab_cropped, 'SLP_simLab_cropped', persistImages=True)
        display_depth_image_with_colormap(SLP_danaLab_cropped, 'SLP_danaLab_cropped', persistImages=True)

        # display histograms
        bins = 200
        display_depth_histogram(SLP_simLab_cropped, 'SLP_simLab_cropped', bins=bins)
        display_depth_histogram(SLP_danaLab_cropped, 'SLP_danaLab_cropped', bins=bins)


    # Wait for any key press to close the window
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return


def readandCropSLPDepthImg(relativePath: str, cropParams_scaleY_x_y = None):
    '''
    Subj 1 simLab: cropParams_scale_x_y = [0.67, 0., 0.07] #[heightScale, shiftX, shiftY]
    '''
    img = io.imread(relativePath) # read the SLP image
    npImg =  np.array(img) # convert to numpy array
    if (cropParams_scaleY_x_y == None):
        cropped, newMinY, newMinX = cropDepthPngFromSimLab(npImg, subj=1) # crop image (hardcode subj 1 cropping) #? use for simlab inputs
    else:
        cropped, newMinY, newMinX = cropNpImage_scaleY_XY(npImg,
                                                          cropParams_scaleY_x_y[0],
                                                          cropParams_scaleY_x_y[1],
                                                          cropParams_scaleY_x_y[2])
    return cropped


def save_array_to_csv(array, filename="depth_image.csv"):
    '''Helper function to save numpy array to .csv for inspection'''
    if array.ndim != 2: # Ensure the input is a 2D NumPy array
        raise ValueError("Input array must be 2D.")
    # Save the array to a CSV file
    np.savetxt('D455_testing/'+filename, array, delimiter=",", fmt="%d")
    print(f"Array saved as root/D455_testing/{filename}")


def display_depth_histogram(depth_image, name: str, bins=10):
    '''plot histagram of depth image'''
    if depth_image.ndim != 2: # Ensure the input is a 2D NumPy array
        raise ValueError("Input depth image must be a 2D array.")
    plt.hist(depth_image.ravel(), bins=bins, edgecolor='black') # Plot the histogram with 10 bins
    # Add labels and title for clarity
    plt.xlabel('Pixel Value'); plt.ylabel('Frequency')
    plt.title('Depth Image Histogram (10 Bins) for img: '+ name)
    plt.show() # Show the plot


def display_cumulative_distribution(depth_image, name: str):
    depth_values = depth_image.flatten() # Flatten the depth image to a 1D array
    pixel_counts = np.bincount(depth_values, minlength=256) # Count occurrences of each pixel value (0 to 255)
    cumulative_distribution = np.cumsum(pixel_counts) # Compute the cumulative distribution
    cumulative_distribution = cumulative_distribution / cumulative_distribution[-1] # Normalize the cumulative distribution (divide by the total number of pixels)
    
    # Plot the cumulative distribution
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(256), cumulative_distribution, color='blue')
    plt.title("Cumulative Distribution of Depth Image (0-255) for img: "+name)
    plt.xlabel("Pixel Value")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.show()


def threshold_depth_image(depth_image, threshold_value, background_value: int = None):
    """
    Set all areas with pixel values greater than a given threshold
    value to the maximum value present in the image
    - params
        - depth_image (np.ndarray): A 2D numpy array representing the depth image in uint16.
        - threshold_value
    
    - Returns
        - np.ndarray: The modified depth image.
    """
    # Ensure the image is of type uint16
    assert depth_image.dtype == np.uint16, "The input depth image must be of type uint16"
    
    mask_value = background_value if(background_value != None) else depth_image.max()
    
    # Set all values greater than the threshold to the maximum value
    new_depth_image = depth_image.copy()
    new_depth_image[depth_image > threshold_value] = mask_value
    
    return new_depth_image

def set_margin_to_max(image, margin_percent_x, margin_percent_y):
    """
    Set all pixels within a percentage margin of the x and y edges to the highest value in the image.
    - params
        - image (np.ndarray): A 2D numpy array representing the image.
        - margin_percent_x (float): The percentage margin for the x-axis (width).
        - margin_percent_y (float): The percentage margin for the y-axis (height).
    - returns
        - np.ndarray: The modified image.
    """
    # Find the maximum value in the image
    max_value = image.max()
    
    # Get the image dimensions
    height, width = image.shape[:2]
    
    # Calculate the number of pixels corresponding to the margin percentage for x and y
    margin_x = [int(margin_percent_x[0] * width), int(margin_percent_x[1] * width)]
    margin_y = [int(margin_percent_y[0] * height), int(margin_percent_y[1] * height)]
    
    # Set the margins to the maximum value
    # Top margin
    image[:margin_y[0], :] = max_value
    if(margin_y[1] != 0):
        # Bottom margin
        image[-margin_y[1]:, :] = max_value
    # Left margin
    image[:, :margin_x[0]] = max_value
    # Right margin
    if(margin_x[1] != 0):
        image[:, -margin_x[1]:] = max_value
    
    return image

def plot_tensor_cumulative_distribution(tensor, name: str):
    # Convert torch tensor to numpy
    img_np = tensor.cpu().numpy()
    img_np_flat = img_np.flatten()

    # Compute and plot cumulative distribution
    sorted_values = np.sort(img_np_flat)
    cumulative = np.cumsum(sorted_values)
    cumulative = cumulative / cumulative[-1]  # normalize to 1

    plt.figure(figsize=(6, 4))
    plt.plot(sorted_values, cumulative)
    plt.title('Cumulative Distribution of input tensor: '+name)
    plt.xlabel('Pixel Value')
    plt.ylabel('Cumulative Distribution')
    plt.grid(True)
    plt.show()

def truncate_min_to_percent(npImage, percentile: float):
    # Calculate the lower bound based on percentile
    lower_bound = np.percentile(npImage, 1.0 - percentile)
    # upper_bound = np.percentile(image, 1.0-marginPercentile) #? after masking, this causes errors, only truncate close pixels
    upper_bound = npImage.max()
    
    # Truncate the values outside this range
    truncated_image = np.clip(npImage, lower_bound, upper_bound)
    
    return truncated_image, lower_bound, upper_bound

def truncate_min_to_threshold(npImage, threshold: int):
    upper_bound = npImage.max()
    # Truncate the values outside this range
    truncated_image = np.clip(npImage, threshold, upper_bound)
    return truncated_image

def scale_to_fit_range(npImage: np.ndarray, currentMin: int, currentMax: int, desiredMin: int, desiredMax: int):
    '''
    Scales numpy image uniformly such that pixels which were currentMin and currentMax becomes desiredMin and desiredMax.
    - params:
        - currentMin: [0-255] pixel value to be scaled to reach desiredMin
        - currentMax: [0-255] pixel value to be scaled to reach desiredMax
        - desiredMin: target minimum value after scaling
        - desiredMax: target maximum value after scaling
    '''
    # Avoid division by zero in case currentMin == currentMax
    if (currentMin == currentMax): raise ValueError('scale_to_fit_range: min == max')

    image = npImage.astype(np.int) #! temp convert to signed int

    # Step 1: Rescale to fit desired range
    scale_factor = (desiredMax - desiredMin) / (currentMax - currentMin)
    scaled_img = (image - currentMin) * scale_factor + desiredMin # center scale about min
    
    # Step 2: Clip the result to ensure values are in the valid range [0, 255]
    scaled_img = np.clip(scaled_img, 0, 255)
    scaled_img = scaled_img.astype(np.uint16) #! convert back to original
    
    return scaled_img

def save_depth_as_greyscale_png(npImage, output_path):
    '''
    Save a numpy depth image as a greyscale PNG.
    - params:
        - npImage: numpy array representing the depth image
        - output_path: file path to save the image (e.g., 'output.png')
    '''
    # Ensure the image is in the range 0-255
    if npImage.max() > 255 or npImage.min() < 0:
        # Normalize the image to fit in the range 0-255
        npImage = cv2.normalize(npImage, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 to save as PNG in greyscale
    npImage_uint8 = npImage.astype(np.uint8)

    # Save the image using OpenCV
    cv2.imwrite(output_path, npImage_uint8)

def apply_mask_with_line(npImage, point, angle_degrees, maskBelow=False):
    '''
    Masks all pixel values in the image above the line defined by the given point and angle.\n
    Pixels above the line are set to the maximum present in the image.
    
    - npImage: 2D numpy array representing the image
    - point: tuple of (x, y) pixel coordinates in the image
    - angle_degrees: angle of the line in degrees, measured counterclockwise from the horizontal axis
    '''
    height, width = npImage.shape
    max_value = np.max(npImage)
    
    angle_radians = math.radians(angle_degrees) # Convert angle from degrees to radians
    #? NOTE slope flipped to make it less confusing since (0,0) is top left
    slope = - math.tan(angle_radians) # Calculate the slope (dy/dx) of the line
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij') # Create a meshgrid of pixel coordinates
    
    # Determine whether each pixel is above the line (y > mx + b)
    y_line = slope * (x_indices - point[0]*width) + point[1]*height
    if(maskBelow):
        mask = y_indices > y_line  # mask pixels below the line
    else:
        mask = y_indices < y_line  # mask pixels above the line

    npImage[mask] = max_value # Apply the mask, setting the masked pixels to the maximum value in the image
    
    return npImage

def apply_circular_mask(npImage, center, radius_ratio):
    '''
    Masks all pixel values outside a circular region defined by the center and radius.\n
    Pixels outside the circle are set to the maximum present in the image.
    
    - npImage: 2D numpy array representing the image
    - center: tuple of (x, y) coordinates for the center of the circle, normalized from 0 to 1
    - radius_ratio: radius of the circle as a proportion of the image height (0 to 1)
    '''
    # Calc centre and radius in pixel coordinates
    height, width = npImage.shape 
    center_x = center[0] * width; center_y = center[1] * height # centre
    radius = radius_ratio * height # radius
    
    # Create a meshgrid of pixel coordinates
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Calculate the distance of each pixel from the center of the circle
    distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    mask = distance_from_center < radius # create mask for pixels inside radius
    npImage[mask] = np.max(npImage) # set masked pixels to max value in image
    
    return npImage

if __name__ == '__main__':
    main()