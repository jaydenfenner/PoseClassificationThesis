'''
This script runs inference on D455_S01 images and saves heatmaps to the same directory
- No functional changes compared to V3 or V4 scripts
'''

import os
import torch
import numpy as np
from inferenceUtils.constants import constants, CoverType, PretrainedModels
from inferenceUtils.loadImage import cropAndRotate_D455, readD455DepthRaw, prepareNpDepthImgForInference
from inferenceUtils.croppingSimLab import cropDepthPngFromSimLab
# from utils.utils_ds import get_max_preds #? removed after replacing with single-image version 
import utils.vis as vis
import cv2
from inferenceUtils.modelUtils import loadPretrainedModel, getModelProperties, getPredsFromHeatmaps

from D455_testing.comparePreprocessing import display_depth_image_with_colormap, displayTorchImg, readandCropSLPDepthImg, threshold_depth_image, display_depth_histogram
from D455_testing.comparePreprocessing import set_margin_to_max, truncate_min_to_threshold, scale_to_fit_range, apply_mask_with_line, apply_circular_mask

from inferenceUtils.imageUtils import scaleNpImageTo255

def main():
    modelType = PretrainedModels.HRPOSE_DEPTH # hardcoded to only use depth u12 model for now
    modelProperties = getModelProperties(modelType=modelType)
    
    experiment_number = 'V3.0'
    modelName = modelProperties.pretrained_model_name
    # save_path = os.path.join('testImgOutput', modelName, plottedImgFolderName) # TODO change back to original
    save_path = os.path.join('Preds_S01_randomMaskPillow')
    # make save directory
    # if not os.path.exists(save_path): os.makedirs(save_path)
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
        os.makedirs(f'{save_path}/no_pillow')
        os.makedirs(f'{save_path}/with_pillow')
    
    print(f"--------------- Running Inference Script ---------------")
    print(f"model:              {modelName}")
    print(f"experiment_number:  {experiment_number}")
    print()

    print(f"        Progress: " + '_'*45) # TODO change this to number of images
    print(f"                : ", end='')

    # TODO read some metadata file to retrieve cropping insets, but just hardcode for now
    # yInset, xInset = readMetadataFile ...

    #! Load and prep the model
    model = loadPretrainedModel(modelType=modelType)
    model.eval() # switch to evaluate mode


    ################ image filenames list here #########################
    if (True):
        poseTuples = [
            ('BACK', 'B'),
            ('RIGHT', 'R'),
            ('FRONT', 'F'),
            ('LEFT_(HARD)', 'LH'),
        ]
        covers = ['0', 'S-60', 'S-90', 'Q-60', 'Q-90']
        pillows = ['np', 'wp']
    else:
        poseTuples = [('BACK', 'B')]; covers = ['Q-90']; pillows = ['wp']
        

    files = []
    for poseTuple in poseTuples:
        for pillow in pillows:
            for cover in covers:
                sampleName = f"S01_{poseTuple[1]}_{cover}_{pillow}"
                noCoverSampleName = f"S01_{poseTuple[1]}_0_{pillow}"
                sampleDirPath = os.path.join("D455_S01", poseTuple[0])
                files.append([sampleName, noCoverSampleName, sampleDirPath])
            

    # TODO read and index all of the input images
    for file in files:
        print(f"#", end='', flush=True)
        print() # TODO remove

        sampleName = file[0]
        noCoverSampleName = file[1]
        sampleDirPath = file[2]
        depthRawPath = os.path.join(sampleDirPath, sampleName+'_Depth.raw')
        colorImgPath = os.path.join(sampleDirPath, sampleName+'_Color.png')
        color_noCover_Path = os.path.join(sampleDirPath, noCoverSampleName+'_Color.png')

        fullPredsPath = None
        if (sampleName.split('.')[0].split('_')[-1] == 'np'):
            fullPredsPath = os.path.join(save_path, 'no_pillow')
        elif (sampleName.split('.')[0].split('_')[-1] == 'wp'):
            fullPredsPath = os.path.join(save_path, 'with_pillow')

        # TODO SLP vs .raw read/crop depending on filename extension?

        #! read and crop RGB version
        bgr_D455_orig = cv2.imread(colorImgPath, cv2.IMREAD_COLOR)
        bgr_D455_cropped = cropAndRotate_D455(bgr_D455_orig, runName='D455_S01') #? note run name must match option in cropAndRotate_D455

        #! read and crop noCover version
        bgr_D455__noCover_orig = cv2.imread(color_noCover_Path, cv2.IMREAD_COLOR)
        bgr_D455__noCover_cropped = cropAndRotate_D455(bgr_D455__noCover_orig, runName='D455_S01') #? note run name must match option in cropAndRotate_D455

        #! read and crop depth version
        depth_D455_orig = readD455DepthRaw(depthRawPath) # read image - pre-crop
        depth_D455_cropped = cropAndRotate_D455(depth_D455_orig, runName='D455_S01') #? note run name must match option in cropAndRotate_D455
        depth_D455_cropped = scaleNpImageTo255(depth_D455_cropped, suppressScaleWarning=True)

        depth_D455_cropped = apply_mask_with_line(depth_D455_cropped, point=(0.81, 0.5), angle_degrees=89, maskBelow=True)
        depth_D455_cropped = apply_circular_mask(depth_D455_cropped, center=(0.903, 0.21), radius_ratio=0.103) # top leg
        depth_D455_cropped = apply_circular_mask(depth_D455_cropped, center=(0.889, 0.68), radius_ratio=0.102) # bottom leg

        #? scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
        depth_D455_cropped = scale_to_fit_range(depth_D455_cropped, 
                                          currentMin=202, currentMax=234, # current based on histogram before scaling
                                          desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
        depth_D455_cropped = threshold_depth_image(depth_D455_cropped, threshold_value=182, background_value=220)

        #! additional maksing to remove pillow entirely (try to prevent preds on pillow)
        # depth_D455_cropped = apply_mask_with_line(depth_D455_cropped, point=(0.5, 0.2), angle_degrees=0, maskBelow=False) #! THIS DOES NOT WORK AT ALL 
        depth_D455_cropped = apply_synthetic_occlusion(depth_D455_cropped, heightP=0.15, widthP=0.4, topLeftP=[0.35, 0.01])

        cropped_depth = depth_D455_cropped
        cropped_color = bgr_D455_cropped
        cropped_color_noCover = bgr_D455__noCover_cropped
        
        input_img = prepareNpDepthImgForInference(cropped_depth, modelType=modelType) # convert to tensor and normalise
        # displayTorchImg(input_img, 'D455-input', persistImages=True) #TODO maybe write something to save these for future reference
        input = input_img.unsqueeze(0) # Add batch dimension to make it [1, channels, height, width]

        #! run inference
        with torch.no_grad(): # prevent gradient calculations
            output = model(input) # (with batches) #? shape --> torch.Size([1, 14, 64, 64])
            heatmaps = output.squeeze().numpy() # convert output to numpy heatmaps, squeeze to remove batch dimension

        preds_raw_64 = getPredsFromHeatmaps(heatmaps) # simple argmax on each heatmap plus masking for negative values

        # display_heatmap(heatmap=heatmaps[10], name=f"L_elbow_{sampleName}") #! L elbow
        # display_heatmap(heatmap=heatmaps[11], name=f"L_wrist_{sampleName}") #! L wrist
        # display_heatmap(heatmap=heatmaps[0], name=f"R_ankle_{sampleName}") #! R ankle
        heatmapPath = f"{fullPredsPath}/{sampleName}_heatmaps"
        if not os.path.exists(heatmapPath): os.makedirs(heatmapPath)
        for i, heatmap in enumerate(heatmaps):
            save_heatmap(heatmap, f'{heatmapPath}/{constants.joints_name[i]}.png')

        print(f'preds: {preds_raw_64}')

        # scale predictions (currently [64,64] to input image size [256, 256] #! note different models will have different sizes
        preds_scaled_depth = np.ones((preds_raw_64.shape[0], 3)) # incude visibility flag = 1 (true for all)
        preds_scaled_depth[:,:2] = preds_raw_64 / heatmaps.shape[1] * cropped_depth.shape[0] #! map preds to input image coords (from [0-64] to orig pixels)

        preds_scaled_color = np.ones((preds_raw_64.shape[0], 3)) # incude visibility flag = 1 (true for all)
        preds_scaled_color[:,:2] = preds_raw_64 / heatmaps.shape[1] * bgr_D455_cropped.shape[0] #! map preds to input image coords (from [0-64] to orig pixels)
        # print(f"fullScale_img.shape: {fullScale_img.shape}") #! (384, 384)

        #! plot preds and gts onto cropped image
        cropped_depth_cv2 = scaleNpImageTo255(cropped_depth) # scale to 0-255 and convert to uint8
        depth_preds_vis = scale_and_shift_255(cropped_depth_cv2, scaleFactor=10, subjPeak=171)
        # display_depth_histogram(depth_preds_vis, 'depth_vis_histogram', bins=200)
        depth_preds_vis = depth_preds_vis.astype(np.uint8)
        depth_preds_vis = cv2.applyColorMap(depth_preds_vis, cv2.COLORMAP_BONE) # get image in rgb (h, w, 3)
        depth_preds_vis = vis.vis_keypoints(depth_preds_vis, preds_scaled_depth, kps_lines=constants.skels_idx) # plot preds

        cropped_color_cv2 = scaleNpImageTo255(cropped_color) # scale to 0-255 and convert to uint8
        cropped_color_cv2 = cropped_color_cv2.astype(np.uint8)
        color_preds_vis = cropped_color_cv2.copy() #! cv2.line throws type error if you don't do this... something about being column vs row indexed
        # color_preds_vis = plot_pck_filled_circles(color_preds_vis, preds_scaled_color, head_proportion=0.105)
        color_preds_vis = vis.vis_keypoints(color_preds_vis, preds_scaled_color, kps_lines=constants.skels_idx) # plot pred

        color_noCover_preds_vis = scaleNpImageTo255(cropped_color_noCover) # scale to 0-255 and convert to uint8
        color_noCover_preds_vis = color_noCover_preds_vis.astype(np.uint8)
        color_noCover_preds_vis = color_noCover_preds_vis.copy() #! cv2.line throws type error if you don't do this... something about being column vs row indexed
        color_noCover_preds_vis = plot_pck_filled_circles(color_noCover_preds_vis, preds_scaled_color, head_proportion=0.105)
        color_noCover_preds_vis = vis.vis_keypoints(color_noCover_preds_vis, preds_scaled_color, kps_lines=constants.skels_idx) # plot pred

        #! save plotted image
        # img_name = f'preds_{img[1]}_{img[2]}' # TODO create unique name for image (based off original?)
        cv2.imwrite(f"{fullPredsPath}/{sampleName}.jpg", depth_preds_vis)
        cv2.imwrite(f"{fullPredsPath}/{sampleName}_color.jpg", color_preds_vis)
        cv2.imwrite(f"{fullPredsPath}/{sampleName}_color_noCover.jpg", color_noCover_preds_vis)

    print()

    # # Wait for any key press to close the window
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    # return



def display_heatmap(heatmap, name: str):
    '''
    Display a heatmap
    - heatmap: 2D numpy array of shape (64, 64), dtype float32
    '''
    assert isinstance(heatmap, np.ndarray), 'heatmap should be numpy.ndarray'
    assert heatmap.ndim == 2, 'heatmap should be 2-ndim'
    
    # Normalize heatmap to 0-255 range for visualization
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap for better visualization
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Display the heatmap
    cv2.imshow(name, heatmap_colored)
    # cv2.waitKey(0)  # Wait for any key press
    # cv2.destroyAllWindows()  # Close the window after the key press

def save_heatmap(heatmap, path):
    '''
    Save a heatmap to a file.
    - save_path: optional file path to save the image
    '''
    assert isinstance(heatmap, np.ndarray), 'heatmap should be numpy.ndarray'
    assert heatmap.ndim == 2, 'heatmap should be 2-ndim'
    
    # Normalize heatmap to 0-255 range for visualization
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap for better visualization
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Save the heatmap if save_path is provided
    if path:
        cv2.imwrite(path, heatmap_colored)


def plot_pck_filled_circles(image, predictions, head_proportion, color=(0, 255, 0), alpha=0.1):
    '''
    Plots filled circles around predictions on an image, with adjustable transparency (alpha).
    - image: the input RGB image as a numpy array
    - predictions: 14x2 array containing the (x, y) coordinates of predicted keypoints
    - head_proportion: the proportion of the image height representing the head size (used to determine the radius)
    - color: the color of the circles (default is green)
    - alpha: transparency of the filled circles (default is 0.5)
    
    - returns: the image with filled circles drawn on it
    '''
    # Ensure alpha is between 0 and 1
    alpha = np.clip(alpha, 0, 1)
    
    height, width, _ = image.shape
    radius = int(head_proportion * height * 0.5)  # Radius is 50% of the head size
    
    # Create an overlay image for blending
    overlay = image.copy()
    
    # Iterate through each predicted keypoint
    for [x, y, vis] in predictions:
        # Draw a filled circle at each prediction on the overlay image
        center = (int(x), int(y))  # Convert (x, y) coordinates to integers
        cv2.circle(overlay, center, radius, color, -1, lineType=cv2.LINE_AA)  # -1 indicates filled circle
        cv2.circle(image, center, radius, (0,0,0), 1, lineType=cv2.LINE_AA)  # -1 indicates filled circle
    
    # Blend the overlay with the original image using alpha transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image

def scale_and_shift_255(npImage: np.ndarray, scaleFactor: float, subjPeak: int):
    '''scale uint8 numpy image by scaleFactor and centre subject in scale'''
    image = npImage.astype(np.int) #! temp convert to signed int

    scaled_img = (image - subjPeak) * scaleFactor + 128
    
    # Step 2: Clip the result to ensure values are in the valid range [0, 255]
    scaled_img = np.clip(scaled_img, 0, 255)
    scaled_img = scaled_img.astype(np.uint8) #! convert back to original
    
    return scaled_img

def apply_synthetic_occlusion(npImage: np.ndarray, heightP, widthP, topLeftP = [0.5, 0.5]):
    height, width = npImage.shape
    xmin = int(topLeftP[0] * width)
    ymin = int(topLeftP[1] * height)
    h = int(height * heightP)
    w = int(width * widthP)
    npImage[ymin:ymin + h, xmin:xmin + w] = np.random.rand(h, w) * 255
    return npImage.copy()

if __name__ == '__main__':
    main()




# not sure when this happened but not deleting bc murphy's law...
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 64 1 3 3, but got 3-dimensional input of size [1, 384, 384] instead
