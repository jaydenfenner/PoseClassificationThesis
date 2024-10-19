'''
This script runs initial (quick and dirty) inference on D455_S01 images
- No functional changes compared to V3 script
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

from D455_testing.comparePreprocessing import display_depth_image_with_colormap, displayTorchImg, readandCropSLPDepthImg, threshold_depth_image
from D455_testing.comparePreprocessing import set_margin_to_max, truncate_min_to_threshold, scale_to_fit_range, apply_mask_with_line, apply_circular_mask

from inferenceUtils.imageUtils import scaleNpImageTo255

def main():
    modelType = PretrainedModels.HRPOSE_DEPTH # hardcoded to only use depth u12 model for now
    modelProperties = getModelProperties(modelType=modelType)
    
    experiment_number = 'V3.0'
    modelName = modelProperties.pretrained_model_name
    # save_path = os.path.join('testImgOutput', modelName, plottedImgFolderName) # TODO change back to original
    save_path = os.path.join('quickResutlts_D455_S01')
    # make save directory
    if not os.path.exists(save_path): os.makedirs(save_path)
    
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
    poseTuples = [
        ('BACK', 'B'),
        ('RIGHT', 'R'),
        ('FRONT', 'F'),
        ('LEFT_(HARD)', 'LH'),
    ]
    covers = ['0', 'S-60', 'S-90', 'Q-60', 'Q-90']
    pillows = ['np', 'wp']

    files = []
    for poseTuple in poseTuples:
        for pillow in pillows:
            for cover in covers:
                fileName = f"S01_{poseTuple[1]}_{cover}_{pillow}"
                filePath = os.path.join("D455_S01", poseTuple[0], fileName+'_Depth.raw')
                files.append([fileName, filePath])

    # TODO read and index all of the input images
    for file in files:
        print(f"#", end='', flush=True)
        print() # TODO remove

        # TODO SLP vs .raw read/crop depending on filename extension?

        orig_img = readD455DepthRaw(file[1]) # read image - pre-crop
        D455_cropped = cropAndRotate_D455(orig_img, runName='D455_S01') #? note run name must match option in cropAndRotate_D455
        D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=True)

        D455_cropped = apply_mask_with_line(D455_cropped, point=(0.81, 0.5), angle_degrees=89, maskBelow=True)
        D455_cropped = apply_circular_mask(D455_cropped, center=(0.903, 0.21), radius_ratio=0.103) # top leg
        D455_cropped = apply_circular_mask(D455_cropped, center=(0.889, 0.68), radius_ratio=0.102) # bottom leg

        #? scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
        D455_cropped = scale_to_fit_range(D455_cropped, 
                                          currentMin=202, currentMax=234, # current based on histogram before scaling
                                          desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
        D455_cropped = threshold_depth_image(D455_cropped, threshold_value=182, background_value=220)
        cropped_img = D455_cropped
        
        input_img = prepareNpDepthImgForInference(cropped_img, modelType=modelType) # convert to tensor and normalise
        # displayTorchImg(input_img, 'D455-input', persistImages=True) #TODO maybe write something to save these for future reference
        input = input_img.unsqueeze(0) # Add batch dimension to make it [1, channels, height, width]

        #! run inference
        with torch.no_grad(): # prevent gradient calculations
            output = model(input) # (with batches) #? shape --> torch.Size([1, 14, 64, 64])
            heatmaps = output.squeeze().numpy() # convert output to numpy heatmaps, squeeze to remove batch dimension

        preds = getPredsFromHeatmaps(heatmaps) # simple argmax on each heatmap plus masking for negative values
        print(f'preds: {preds}')

        # scale predictions (currently [64,64] to input image size [256, 256] #! note different models will have different sizes
        pred2d_cropped = np.ones((preds.shape[0], 3)) # incude visibility flag = 1 (true for all)
        pred2d_cropped[:,:2] = preds / heatmaps.shape[1] * cropped_img.shape[0] #! map preds to input image coords (from [0-64] to orig pixels)
        # print(f"fullScale_img.shape: {fullScale_img.shape}") #! (384, 384)

        #! plot preds and gts onto cropped image
        cropped_img_cv2 = scaleNpImageTo255(cropped_img) # scale to 0-255 and convert to uint8
        cropped_img_cv2 = cropped_img_cv2.astype(np.uint8)
        img_patch_vis = cv2.applyColorMap(cropped_img_cv2, cv2.COLORMAP_BONE) # get image in rgb (h, w, 3)
        tmpimg = vis.vis_keypoints(img_patch_vis, pred2d_cropped, kps_lines=constants.skels_idx) # plot preds

        #! save plotted image
        # img_name = f'preds_{img[1]}_{img[2]}' # TODO create unique name for image (based off original?)
        if (file[0].split('.')[0].split('_')[-1] == 'np'):
            cv2.imwrite(os.path.join(save_path, 'no_pillow', file[0]+'.jpg'), tmpimg)
        elif (file[0].split('.')[0].split('_')[-1] == 'wp'):
            cv2.imwrite(os.path.join(save_path, 'with_pillow', file[0]+'.jpg'), tmpimg)

    print()

    # Wait for any key press to close the window
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# not sure when this happened but not deleting bc murphy's law...
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 64 1 3 3, but got 3-dimensional input of size [1, 384, 384] instead
