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
from D455_testing.comparePreprocessing import display_depth_image_with_colormap, displayTorchImg, readandCropSLPDepthImg
from inferenceUtils.imageUtils import scaleNpImageForOpencv

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
        ['D455_testing/D455_S01_u.raw', 'D455'],
        ['D455_testing/SLP_subj01_u_p014.png', 'SLP'],
                ]:
        print(f"#", end='', flush=True)
        print() # TODO remove

        # TODO SLP vs .raw read/crop depending on filename extension?

        #! read and prep image
        if img[1] == 'D455':
            orig_img = readD455DepthRaw(img[0]) # read image - pre-crop
            cropped_img = cropAndRotate_D455(orig_img) # rotate 90 deg anticlockwise and crop square region containing bed
            # display_depth_image_with_colormap(cropped_img,'cropped', persistImages=True) ############## diplay for debugging #TODO maybe write something to save these for future reference
        elif img[1] == 'SLP':
            cropped_img = readandCropSLPDepthImg('D455_testing/SLP_subj01_u_p014.png')
            pass
        else:
            raise ValueError("Unknown image type")
        
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
        cropped_img_cv2 = scaleNpImageForOpencv(cropped_img) # scale to 0-255 and convert to uint8
        img_patch_vis = cv2.applyColorMap(cropped_img_cv2, cv2.COLORMAP_BONE) # get image in rgb (h, w, 3)
        tmpimg = vis.vis_keypoints(img_patch_vis, pred2d_cropped, kps_lines=constants.skels_idx) # plot preds

        #! save plotted image
        img_name = 'preds_'+img[1] # TODO create unique name for image (based off original?)
        cv2.imwrite(os.path.join(save_path, img_name+'.jpg'), tmpimg)

    print()

    # Wait for any key press to close the window
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# not sure when this happened but not deleting bc murphy's law...
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 64 1 3 3, but got 3-dimensional input of size [1, 384, 384] instead
