# Using HRPose to run inference on some images and display results

import os
import torch
import numpy as np
from model.HRpose import get_pose_net
from inferenceUtils.constants import constants, CoverType #? numberOfJoints
from inferenceUtils.loadImage import readDepthPngFromSimLab, prepareNpDepthImgForInference
from inferenceUtils.croppingSimLab import cropDepthPngFromSimLab
# from utils.utils_ds import get_max_preds #? removed after replacing with single-image version 
import utils.vis as vis
import cv2

import scipy.io as sio

def getSimLabDepthGTs(subj):
    joints_gt_RGB_t = sio.loadmat(os.path.join("SLP/simLab", '{:05d}'.format(subj), 'joints_gt_RGB.mat'))['joints_gt'] # 3 x n_jt x n_frm -> n_jt x 3
    # print('joints gt shape', joints_gt_RGB_t.shape) #! joints gt shape (3, 14, 45)
    joints_gt_RGB_t = joints_gt_RGB_t.transpose([2, 1, 0])
    joints_gt_RGB_t = joints_gt_RGB_t - 1  # to 0 based # note third dim seems unised as all values are = -1.0
    #! the third value is used in plotting, set all to 1:
    # print(joints_gt_RGB_t[0]) #! [[ 2.55719178e+02  1.20155479e+03 -1.00000000e+00], ... ]
    joints_gt_RGB_t[:,:,2] = 1
    # print('joints gt shape', joints_gt_RGB_t.shape) #! joints gt shape (45, 14, 3)
    # print(joints_gt_RGB_t[0]) #! [[ 2.55719178e+02  1.20155479e+03 1.00000000e+00], ... ]

    # load pointers for base modality (RGB) and target modality (depth)
    pth_PTr_depth = os.path.join("SLP/simLab", '{:05d}'.format(subj), 'align_PTr_depth.npy')
    PTr_depth = np.load(pth_PTr_depth)
    pth_PTr_RGB = os.path.join("SLP/simLab", '{:05d}'.format(subj), 'align_PTr_RGB.npy')
    PTr_RGB = np.load(pth_PTr_RGB)

    # homography RGB to depth #! stolen from SLP (no idea how it works but it converts RGB_gts to depth_gts)
    PTr_RGB2depth = np.dot(np.linalg.inv(PTr_depth), PTr_RGB)
    PTr_RGB2depth = PTr_RGB2depth / np.linalg.norm(PTr_RGB2depth)
    joints_gt_depth_t = np.array(list(
    map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB2depth)[0], joints_gt_RGB_t[:, :, :2])))
    joints_gt_depth_t = np.concatenate([joints_gt_depth_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)
    # print('joints gt_depth shape', joints_gt_depth_t.shape) #! joints gt shape (45, 14, 3)
    # print(joints_gt_depth_t)
    return joints_gt_depth_t

def main():
    experiment_number = 'V2.0'
    modelName = constants.pretrained_model_name
    # subjects_list = range(1,8) # [1, 2, ... , 7] # TODO restore to original
    subjects_list = [7]
    covers_list = [CoverType.COVER1, CoverType.COVER2, CoverType.UNCOVER] # TODO restore to original
    
    print(f"--------------- Running Inference Script ---------------")
    print(f"model:              {modelName}")
    print(f"experiment_number:  {experiment_number}")
    print(f"subjects_list:      {subjects_list}")
    print(f"covers_list:        {[cover.value for cover in covers_list]}")
    print()

    model = loadPretrainedModel()
    model.eval() # switch to evaluate mode
    
    for subj in subjects_list:
        print(f"subj = {subj} of {subjects_list}")
        joints_gt_depth_all = getSimLabDepthGTs(subj=subj) # (45, 14, 3)
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
                input_img = prepareNpDepthImgForInference(cropped_img) # convert to tensor and normalise
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


def getPredsFromHeatmaps(heatmaps):
    '''get predictions in pixel coords from heatmaps'''
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 3, 'heatmaps should be 3-ndim'
    
    preds = np.zeros((14, 2), dtype=np.float32) # initialised to 0 for masking
    for i, heatmap in enumerate(heatmaps):
        idx = np.argmax(heatmap) # Find the index of the maximum value
        y, x = divmod(idx, heatmap.shape[1]) # Convert the index to 2D coordinates
        
        # Apply confidence masking
        if (heatmap[y, x] > 0.0): preds[i] = [x, y] # update preds only if max condfidence > 0.0 (replicate original code)

    return preds

def loadPretrainedModel():
    # get model (architecture only)
    model = get_pose_net(in_ch=constants.numberOfChannels, out_ch=constants.numberOfJoints) # 1 channel (depth), 14 joints

    pretrained_model_path = os.path.join("pretrainedModels", constants.pretrained_model_name, 'model_dump/checkpoint.pth')
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu')) #! note extra argument to use cpu
    model.load_state_dict(checkpoint['state_dict'])

    # model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids) # paralellise the torch operations #! removed since I'm using cpu only
    model = model.to('cpu') # send model to cpu

    return model

if __name__ == '__main__':
    main()


# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 64 1 3 3, but got 3-dimensional input of size [1, 384, 384] instead
