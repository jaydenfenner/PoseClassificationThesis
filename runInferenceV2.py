# Using HRPose to run inference on some images and display results

import os
import torch
import numpy as np
from inferenceUtils.constants import constants, CoverType, PretrainedModels
from inferenceUtils.loadImage import readDepthPngFromSimLab, prepareNpDepthImgForInference
from inferenceUtils.croppingSimLab import cropDepthPngFromSimLab
# from utils.utils_ds import get_max_preds #? removed after replacing with single-image version 
import utils.vis as vis
import cv2
from inferenceUtils.simLabUtils import getSimLabDepthGTsForSubject
from inferenceUtils.modelUtils import loadPretrainedModel, getModelProperties, getPredsFromHeatmaps

def main():
    modelType = PretrainedModels.HRPOSE_DEPTH # hardcoded to only use depth u12 model for now
    modelProperties = getModelProperties(modelType=modelType)

    experiment_number = 'V2.1'
    modelName = modelProperties.pretrained_model_name
    # subjects_list = range(1,8) # [1, 2, ... , 7] # TODO restore to original
    subjects_list = [7]
    covers_list = [CoverType.UNCOVER] #[CoverType.COVER1, CoverType.COVER2, CoverType.UNCOVER] # TODO restore to original
    
    print(f"--------------- Running Inference Script ---------------")
    print(f"model:              {modelName}")
    print(f"experiment_number:  {experiment_number}")
    print(f"subjects_list:      {subjects_list}")
    print(f"covers_list:        {[cover.value for cover in covers_list]}")
    print()

    model = loadPretrainedModel(modelType=modelType)
    model.eval() # switch to evaluate mode
    
    for subj in subjects_list:
        print(f"subj = {subj} of {subjects_list}")
        joints_gt_depth_all = getSimLabDepthGTsForSubject(subj=subj) # (45, 14, 3)
        # print(f"max x or y in gts: {np.max(joints_gt_depth_all)}") #! max x or y in gts: 387.59526927382143

        #* read first image for subject to retrieve cropping insets
        _, yInset, xInset = cropDepthPngFromSimLab(img=readDepthPngFromSimLab(), subj=subj) 
        joints_gt_depth_all[:, :, :2] = joints_gt_depth_all[:, :, :2] - [xInset, yInset] # shift gts to match cropped image

        for cover in covers_list:
            print(f"    cover = {cover.value} of {[cover.value for cover in covers_list]}")
            #! make save directory
            save_dir='testImgOutput'
            plottedImgFolderName = ('-').join(("predsVsGTs", "depthPng", experiment_number))
            save_path = os.path.join(save_dir, modelName, plottedImgFolderName, '{:05d}'.format(subj), cover.value)
            if not os.path.exists(save_path): os.makedirs(save_path)

            print(f"        Progress: " + '_'*45)
            print(f"                : ", end='')
            for poseNum in range(1, 46):
                print(f"#", end='', flush=True)
                #! read and prep image
                orig_img = readDepthPngFromSimLab(subj=subj, cover=cover, poseNum=poseNum)
                cropped_img, yInset, xInset = cropDepthPngFromSimLab(orig_img, subj=subj)
                input_img = prepareNpDepthImgForInference(cropped_img, modelType=modelType) # convert to tensor and normalise
                input = input_img.unsqueeze(0) # Add batch dimension to make it [1, channels, height, width]
                joints_gt_depth = joints_gt_depth_all[poseNum-1] #! (14, 3) --> [x, y, vis]

                # run inference
                with torch.no_grad(): # prevent gradient calculations
                    output = model(input) # (with batches) #! shape --> torch.Size([1, 14, 64, 64])
                    heatmaps = output.squeeze().numpy() #! convert output to numpy heatmaps, squeeze to remove batch dimension

                preds = getPredsFromHeatmaps(heatmaps) #! simple argmax on each heatmap plus masking for negative values

                # scale predictions (currently [64,64] to input image size [256, 256] #! note different models will have different sizes
                pred2d_cropped = np.ones((preds.shape[0], 3)) # incude visibility flag = 1 (true for all)
                pred2d_cropped[:,:2] = preds / heatmaps.shape[1] * cropped_img.shape[0] #! map preds to input image coords (from [0-64] to orig pixels)
                # print(f"fullScale_img.shape: {fullScale_img.shape}") #! (384, 384)

                #! plot preds and gts onto cropped image
                img_patch_vis = cv2.applyColorMap(cropped_img, cv2.COLORMAP_BONE) # get image in rgb (h, w, 3)
                tmpimg = vis.vis_keypoints(img_patch_vis, joints_gt_depth, kps_lines=constants.skels_idx, is_gt=True) # plot gts
                tmpimg = vis.vis_keypoints(tmpimg, pred2d_cropped, kps_lines=constants.skels_idx, is_predForComparison=True) # plot preds

                #! save plotted image
                img_name = ('_').join(('s{:05d}'.format(subj), cover.value, 'p{:02d}'.format(poseNum)))
                cv2.imwrite(os.path.join(save_path, img_name+'.jpg'), tmpimg)
            print()
        print()

if __name__ == '__main__':
    main()


# not sure when this happened but not deleting bc murphy's law...
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 64 1 3 3, but got 3-dimensional input of size [1, 384, 384] instead
