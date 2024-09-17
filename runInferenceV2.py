# Using HRPose to run inference on some images and display results

import os
import torch
import numpy as np
from model.HRpose import get_pose_net
from loadImageCustom import constants #? numberOfJoints
from loadImageCustom import readAndCropDepthPngFromSimLab, preparePngForInference
# from utils.utils_ds import get_max_preds #? removed after replacing with single-image version 
import utils.vis as vis
import utils.utils as ut
import cv2

def main():
    model = loadPretrainedModel()

    model.eval() # switch to evaluate mode

    # TODO define which simLab samples are to be loaded for inference
    # for subj in ......
    subj = 1
    cover = constants.CoverType.UNCOVER
    poseNum = 1

    fullScale_img = readAndCropDepthPngFromSimLab(subj=subj, cover=cover, poseNum=poseNum)
    input_img = preparePngForInference(fullScale_img)
    
    input = input_img.unsqueeze(0) # Add batch dimension to make it [1, channels, height, width]

    # run inference
    with torch.no_grad(): # prevent gradient calculations
        output = model(input) # (with batches) #! shape --> torch.Size([1, 14, 64, 64])
        heatmaps = output.squeeze().numpy() #! convert output to numpy heatmaps, squeeze to remove batch dimension
        preds = getPredsFromHeatmaps(heatmaps) #! simple argmax on each heatmap plus masking for negative values

        # scale predictions (currently [64,64] to input image size [256, 256] #! note different models will have different sizes
        pred2d_patch = np.ones((preds.shape[0], 3)) # incude visibility flag = 1 (true for all)
        pred2d_patch[:,:2] = preds / heatmaps.shape[1] * fullScale_img.shape[0] #! map preds to input image coords (from [0-64] to orig pixels)
        print(f"fullScale_img.shape: {fullScale_img.shape}")

        img_patch_vis = cv2.applyColorMap(fullScale_img, cv2.COLORMAP_BONE) # get image in rgb (h, w, 3)

        '''
        TARGET:
        vis.vis_keypoints --> 
        
        img.shape: (256, 256, 3) #! need to apply colour map but not rescale
        
        kps:    [[112. 240.   1.] #! need to scale back to [0-256]
                [112. 188.   1.]
                [112. 132.   1.]
                [144. 136.   1.]
                [140. 184.   1.]
                [140. 240.   1.]
                [116. 112.   1.]
                [ 92.  92.   1.]
                [108.  56.   1.]
                [156.  60.   1.]
                [164.  92.   1.]
                [144. 108.   1.]
                [128.  52.   1.]
                [124.  16.   1.]]
        '''

        img_patch=img_patch_vis; pred_2d=pred2d_patch; skel=constants.skels_idx; sv_dir='testImgOutput'; suffix='-TEST_SUFFIX'; idx=1

        sv_dir = os.path.join(sv_dir, '2d' + suffix) # make save directory
        if not os.path.exists(sv_dir): os.makedirs(sv_dir)

        # tmpimg = vis.vis_keypoints(img_patch, pred_2d, skel, is_gt=True) # plot gts
        tmpimg = vis.vis_keypoints(img_patch, pred_2d, skel) # plot preds

        cv2.imwrite(os.path.join(sv_dir, str(idx) + '.jpg'), tmpimg)



        # _, avg_acc, cnt, pred_hm = accuracy(output.cpu().numpy(), target.cpu().numpy()) #? where target is the gt_heatmaps
        # def accuracy():
            # pred, _ = get_max_preds(output) #! [N x n_jt x 2] --> per-batch 2D keypoints for each of the 14 joints

            # norm = np.ones((1, 2)) * np.array([256, 256]) / 10  # use 0.1 as norm
            # dists = calc_dists(pred, target, norm)  # given one , so not normalized


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