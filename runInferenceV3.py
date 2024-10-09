'''
This script adds to the functionality of the previous (runInferenceV2) by:
- Take input as any image in the specified directory
    - images should be in format output by D455 "snapshots"
    - (i.e. .raw files, 2 bytes per pixel, width=640, height=360)
    - Each session should be in it's own file with a csv provided containing relevant metadata and cropping inputs

Still using HRpose trained on depth images, cover=u12
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
    plottedImgFolderName = ('-').join(("D455_Inferece", "snapshots", experiment_number))
    # save_path = os.path.join('testImgOutput', modelName, plottedImgFolderName) # TODO change back to original
    save_path = os.path.join('D455_testing')
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

    # TODO read and index all of the input images
    for img in [
        # ['D455_testing/simLab_subj01_u_p014.png', 'simLab', 's01_u'],
        # ['D455_testing/danaLab_sample_u.png', 'danaLab', 's01_u'],
        # ['D455_testing/danaLab_side_sample_u.png', 'danaLab', 's02_u'],
        # ['D455_testing/D455_S01_u.raw', 'D455_V1', 'S01_u'],
        # ['D455_testing/D455_S02_u.raw', 'D455_V1', 'S02_u'],
        # ['D455_testing/D455_S04_c.raw', 'D455_V1', 'S04_c'],
        # ['D455_testing/D455_S07_u.raw', 'D455_V1', 'S07_u'],
        # ['D455_cinema2/300_Depth.raw', 'D455_V3', 'S01_u'],
        # ['D455_cinema2/200_Depth.raw', 'D455_V3', 'S02_u'],
        ['D455_cinema1/1_Depth.raw', 'D455_V2', 'S01_u'], 
        ['D455_cinema1/2_Depth.raw', 'D455_V2', 'S02_u'], 
        ['D455_cinema1/3_Depth.raw', 'D455_V2', 'S03_u'],
        ['D455_cinema1/6_Depth.raw', 'D455_V2', 'S03_c'], 
        ['D455_cinema1/7_Depth.raw', 'D455_V2', 'S01_c'],
                ]:
        print(f"#", end='', flush=True)
        print() # TODO remove

        # TODO SLP vs .raw read/crop depending on filename extension?

        #! read and prep image
        if img[1] == 'D455_V1':
            if(False): #! PREVIOUS METHOD, DID NOT WORK
                orig_img = readD455DepthRaw(img[0]) # read image - pre-crop
                cropped_img = cropAndRotate_D455(orig_img, runName='D455_V1') # rotate 90 deg anticlockwise and crop square region containing bed
                # display_depth_image_with_colormap(cropped_img,'cropped', persistImages=True) ############## diplay for debugging #TODO maybe write something to save these for future reference
                
                # cropped_img = threshold_depth_image(cropped_img, threshold_value=2200) #! also mask high depth pixels
                # cropped_img = set_margin_to_max(cropped_img, margin_percent_x=[0.1, 0.16], margin_percent_y=[0.03, 0.]) #! also mask margins of image
            else:
                D455_image = readD455DepthRaw(img[0]) # read image - pre-crop
                D455_cropped = cropAndRotate_D455(D455_image, runName='D455_V1') #? note run name must match option in cropAndRotate_D455
                D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=True)
                D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0.1, 0.155], margin_percent_y=[0.03, 0.])
                #? scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
                D455_cropped = scale_to_fit_range(D455_cropped, 
                                                currentMin=180, currentMax=207, # current based on histogram before scaling
                                                desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
                D455_cropped = threshold_depth_image(D455_cropped, threshold_value=189, background_value=220)
                cropped_img = D455_cropped

        elif img[1] == 'D455_V2':
            orig_img = readD455DepthRaw(img[0]) # read image - pre-crop
            D455_cropped = cropAndRotate_D455(orig_img, runName='D455_V2') #? note run name must match option in cropAndRotate_D455
            D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=True)
            D455_cropped = apply_mask_with_line(D455_cropped, point=(0.779, 0.5), angle_degrees=88, maskBelow=True)
            D455_cropped = apply_circular_mask(D455_cropped, center=(1.2, 0.675), radius_ratio=0.45)
            #? scale and shift based on histograms to set bed variation and height equal to danaLab (clip top 0-255 at end)
            D455_cropped = scale_to_fit_range(D455_cropped, 
                                            currentMin=201, currentMax=230, # current based on histogram before scaling
                                            desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
            D455_cropped = threshold_depth_image(D455_cropped, threshold_value=185, background_value=220)
            cropped_img = D455_cropped

        elif img[1] == 'D455_V3':
            orig_img = readD455DepthRaw(img[0]) # read image - pre-crop
            D455_cropped = cropAndRotate_D455(orig_img, runName='D455_V3') # rotate 90 deg anticlockwise and crop square region containing bed
            D455_cropped = scaleNpImageTo255(D455_cropped, suppressScaleWarning=True) # convert image to 0-255 for ease of use
            D455_cropped = set_margin_to_max(D455_cropped, margin_percent_x=[0., 0.25], margin_percent_y=[0., 0.]) # remove ladder with margin
            # scale bed to match danaLab
            D455_cropped = scale_to_fit_range(D455_cropped, 
                                            currentMin=204, currentMax=232, # current based on histogram before scaling
                                            desiredMin=162, desiredMax=186) # desired matching danaLab histogram after cropping
            D455_cropped = threshold_depth_image(D455_cropped, threshold_value=196, background_value=220) # threshold background to match danaLab
            cropped_img = D455_cropped

        elif img[1] == 'simLab':
            cropped_img = readandCropSLPDepthImg(img[0]) #? use for simLab sample
        elif img[1] == 'danaLab':
            cropped_img = readandCropSLPDepthImg(img[0], cropParams_scaleY_x_y=[0.7, 0.12, 0.])  #? use for danaLab sample
        else:
            raise ValueError(f"Unknown image type: {img[1]}")
        
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
        img_name = f'preds_{img[1]}_{img[2]}' # TODO create unique name for image (based off original?)
        cv2.imwrite(os.path.join(save_path, img_name+'.jpg'), tmpimg)

    print()

    # Wait for any key press to close the window
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# not sure when this happened but not deleting bc murphy's law...
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 64 1 3 3, but got 3-dimensional input of size [1, 384, 384] instead
