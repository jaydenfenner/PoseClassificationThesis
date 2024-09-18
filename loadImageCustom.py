'''
GOAL OF THIS FILE:

The goal of this file is to replicate the loading of a single depth image from the 
#? simLab/00001/depth/uncover/image_000001.png
image in a suitable format for inference
'''

import numpy as np
import cv2
import torchvision.transforms as transforms
import utils.utils as ut
from data.SLP_RD import SLP_RD
import opt
from skimage import io
from enum import Enum

def main():
    #* Checking image sizing
    poseNums = list(range(1,43,6))
    for i, subj in enumerate(range(1,8)):
        img_input = readAndCropDepthPngFromSimLab(subj=subj, cover=constants.CoverType.UNCOVER, poseNum=poseNums[i])
        displayTorchImg(img_input)

    img_input = readAndCropDepthPngFromSimLab(subj=1, cover=constants.CoverType.UNCOVER, poseNum=34)

def displayTorchImg(img_input):
    img_forVisualisation = ut.ts2cv2(img_input, mean=[constants.mean_depth], std=[constants.std_depth])
    img_forVisualisation = cv2.applyColorMap(img_forVisualisation, cv2.COLORMAP_BONE)
    cv2.imshow('custom window name', img_forVisualisation)
    cv2.waitKey(0)

#! parser.add_argument('--sz_pch', nargs='+', default=(256, 256), type=int, help='input image size, model 288, pix2pix 256')
#? SLP_fd_test = SLP_FD(SLP_rd_test,  opts, phase='test', if_sq_bb=True)

#* (Taken from data reader class)
df_cut=0.03 #! auto defined as 0.03 (margin depth below bed before everything is cut off)
d_bed = 2.264  # as meter #! depth of the bed in the simLab set

# d_bed = self.ds.d_bed #* directly defined above
d_cut = d_bed - df_cut  # around human body center
# self.d_cut = d_cut #* above
# self.df_cut = df_cut #* above

# # special part for A2J #* no uses in repo??
# ct_A2J = 2.9    # the pelvis center
# self.bias_A2S = d_cut - ct_A2J

class constants:
    ''' Constants required when preprocessing images'''
    model_input_size = [256, 256] #! required image size to be fed into the model
    model_output_size = [64, 64] #! output heatmap size of the model
    
    mean_depth = 0.7302197 #! mean and std used for normalisation of depth image
    std_depth = 0.25182092

    class CoverType(Enum):
        COVER1 = 'cover1'
        COVER2 = 'cover2'
        UNCOVER = 'uncover'

    #! required to instantiate model
    numberOfJoints = 14
    numberOfChannels = 1 # depth only
    pretrained_model_name = "SLP_depth_u12_HRpose_exp" # name of the pretrained model folder

    #! taken from dataset reader
    skels_name = (
		# ('Pelvis', 'Thorax'),
		('Thorax', 'Head'),
		('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
		('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
		# ('Pelvis', 'R_Hip'),
		('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
		# ('Pelvis', 'L_Hip'),
		('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)
    joints_name = (
		"R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
		"L_Elbow", "L_Wrist", "Thorax", "Head", "Pelvis", "Torso",
		"Neck") 
    skels_idx = ut.nameToIdx(skels_name, joints_name=joints_name)

def preparePngForInference(img):
    #! resize image to required input size
    newSize = constants.model_input_size[0]
    img_input = cv2.resize(img, (newSize, newSize), interpolation=cv2.INTER_AREA)

    #! convert to datatype supported by pytorch (float in format h,w,c)
    img_input = img_input.astype(np.float32)
    img_input = img_input[..., None]

    #! convert to tensor and normalise
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.mean_depth], std=[constants.std_depth])
        ])
    img_input = torch_transforms(img_input) #! imagePatch_Pytorch

    # displayTorchImg(img_input) #! display image for debugging purposes
    return img_input

def readDepthPngFromSimLab(subj=1, cover: constants.CoverType = constants.CoverType.UNCOVER, poseNum=1):
    '''
    Take an SLP simLab subject, cover and pose number and return the image from simLab\n
    NOTE: subj from 1-7\n
    NOTE: cover from constants.CoverType
    NOTE: poseNum from 1-45\n
    '''
    #! construct path, read depth image png and convert to numpy array
    subjWithZeros = "{:05d}".format(subj)
    poseNumWithZeros = "{:06d}".format(poseNum)
    path_to_depthImg = f'SLP/simLab/{subjWithZeros}/depth/{cover.value}/image_{poseNumWithZeros}.png'
    img = io.imread(path_to_depthImg)
    img = np.array(img)
    return img

def cropDepthPngFromSimLab(img):
    #! crop image to fixed square in centre
    origHeight, origWidth = img.shape[:2]
    shiftPercentageYX = [0.01, 0]
    newSize = int(origHeight * 0.75)
    newMinY = (origHeight - newSize) // 2 + int(origHeight * shiftPercentageYX[0])
    newMinX = (origWidth - newSize) // 2 + int(origWidth * shiftPercentageYX[1])
    img = img[newMinY:newMinY+newSize, newMinX:newMinX+newSize]
    return img, newMinY, newMinX


def Original_readDepthPng(idx:int, cover='u'):
    '''
    Take an SLP image index, find the file and read the raw depth image as a numpy array
    '''
    coverCondition = 'cover1' if (cover == '1') else 'cover2' if (cover == '2') else 'uncover'
    idxWithZeros = "{:05d}".format(idx)

    #! unused - for depth raw ---> NEEDS TO SOMEHOW BE IN RANGE 0-255 
    # path_to_rawDepth = f'SLP/simLab/{idxWithZeros}/depthRaw/{coverCondition}/0{idxWithZeros}.npy' #* note extra zero on npy file
    # img = np.load(path_to_rawDepth)
    # img = np.array(img)
    #! used - for depth png (note clearly some processing/cleanup has occurred)
    path_to_depthImg = f'SLP/simLab/{idxWithZeros}/depth/{coverCondition}/image_0{idxWithZeros}.png' #* note extra zero on png file
    img = io.imread(path_to_depthImg)
    img = np.array(img)

    #! error checking
    print(f"min: {np.min(img)} max: {np.max(img)}")
    print(img)
    print(f"shape: {img.shape}")

    #! ignore original read and read image instead using provided methods
    '''
    opts = opt.parseArgs()
    pretrained_model_name = "SLP_depth_u12_HRpose_exp" # name of the pretrained model folder
    opts.SLP_set = 'simLab'
    opts.mod_src = ['depth']
    opts.cov_li = ['uncover']
    opts.if_bb = True  # not using bb, give ori directly
    opts = opt.aug_opts(opts)  # add necesary opts parameters   to it
    opts.if_flipTest = False
    opt.set_env(opts)
    ds = SLP_RD(opts, phase=opts.test_par)
    # img, joints_ori, bb = ds.get_array_joints(idx, mod='depthRaw', if_sq_bb=True)  # raw depth .npy    # how transform
    img, joints_ori, bb = ds.get_array_joints(idx, mod='depth', if_sq_bb=True)  # preprocessed depth png    # how transform
    # '''

    #! convert to datatype supported by pytorch
    img = img.astype(np.float32)
    img = img[..., None]

    #! try clipping and see if anything changes
    # TODO no change for depth.png but clips too much for depthRaw
    # TODO need to investigate appropriate settings for using realSense depth camera
    # TODO these might be preprocessing steps in these scripts or settings in the camera
    # for i in range(1):
    #     img[:, :, i] = np.clip(img[:, :, i] * [1.0, 1.0, 1.0][i], 0, 255)

    #! apply transforms to produce final image (toTensor and normalise)
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.mean_depth], std=[constants.std_depth])
        ])
    img_forVisualisation = torch_transforms(img) #! imagePatch_Pytorch
    img_forVisualisation = ut.ts2cv2(img_forVisualisation, mean=[constants.mean_depth], std=[constants.std_depth])
    img_forVisualisation = cv2.applyColorMap(img_forVisualisation, cv2.COLORMAP_BONE)

    cv2.imshow('custom window name', img_forVisualisation)
    cv2.waitKey(0)

    return img

''''''
def performRequiredAugs(img, mods=['depth']):
    '''
    Adjust a 2D depth image to meet the requirements to be input into SLP pretrained models
    '''
    #! need to create a square image with the following characteristics:
    # square shape
    # centred on the centre of the ground truth joints (replicate as best possible) #! use bed corners
    # 1.2x the square containing the ground truth joints
    # rescaled to match required input dims (256, 256, 1)
    # converted to a pytorch tensor
    # normalised based on depth mean and std (mean_depth = 0.7302197, std_depth = 0.25182092)

    '''
    # scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False #! from original, basically no transforms
    # img_patch, trans = ut.generate_patch_image(depth, bb, do_flip=false, scale=1.0, rot=0.0, do_occlusion=false, sz_std=[288, 288] )
    # trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, sz_std[0], sz_std[1], scale, rot, inv=False)
    #! above with arguments only scales the image from input size of (y,x)512x424 to 288x288 and can be replaced with the below:
    #? crop the image to the bounding box
    #? resize to match the required input resolution
    img = cv2.resize(img, (288, 288))
    '''

    print(f"depth shape: {depth.shape}")
    print(depth)
    return
''''''

def jt_hm(self, idx):
        '''
        joint heatmap format feeder.  get the img, hm(gaussian),  jts, l_std (head_size)
        :param index:
        :return:
        '''
        mods = self.opts.mod_src #! ['depth']
        n_jt = self.ds.joint_num_ori # use ori joint #! 14
        sz_pch = self.opts.sz_pch #! (256, 256)
        out_shp = self.opts.out_shp[:2] #! [64, 64]
        ds = self.ds
        mod0 = mods[0]
        li_img = [] #! list of images for each modality
        li_mean =[] #! list of means for each modality
        li_std = [] #! list of standard deviations for each modality
        
        #! image shape: (512, 424)
        img, joints_ori, bb = self.ds.get_array_joints(idx, mod=mod0, if_sq_bb=True)  # raw depth    # how transform

        #-------------------------------------------------------------------------------------------
        def get_array_joints(self, idx_smpl=0, mod='depthRaw', if_sq_bb=True):
            '''
            index sample function in with flattened order with given modalities. It could be raw form array or image, so we call it array.
            corresponding joints will be returned too.  depth and PM are perspective transformed. bb also returned
            :param idx_smpl: the index number base 0
            :param mod:   4 modality including raw data. PM raw for real pressure data
            :return:
            '''
            id_subj, cov, id_frm = self.pthDesc_li[idx_smpl]    # id for file , base 1 #! subject_number_outof_7, cover_condition, pose_index_outof_45

            arr = getImg_dsPM(dsFd=self.dsFd, idx_subj=id_subj, modality=mod, cov=cov, idx_frm=id_frm) #! get the image as an array for the specific subject/cover/pose combo

            # get unified jt name, get rid of raw extenstion
            if 'depth' in mod:
                mod = 'depth'
            mod = uni_mod(mod)  # to unify name for shared annotation #! just becomes mod = 'depth'

            joints_gt = getattr(self, 'li_joints_gt_{}'.format(mod)) #! get joints_groundTruths_list for the entire set for this modality ('depth')
            jt = joints_gt[id_subj-1][id_frm-1] #! get joints_groundtruths for this image (and this modality)

            if if_sq_bb:    # give the s
                bb = getattr(self, 'li_bb_sq_{}'.format(mod))[id_subj - 1][id_frm - 1]
            else:
                bb = getattr(self, 'li_bb_{}'.format(mod))[id_subj-1][id_frm-1]
            return arr, jt, bb
        #-------------------------------------------------------------------------------------------
        
        joints_ori = joints_ori[:n_jt, :2]  # take only the original jts #! before [14,3] -> after [14,2] (probably just a visibility flag or other metadata)

        img_height, img_width = img.shape[:2]       #first 2 #! only 2 values anyways but extract width + height

        #! mean and std from dataset, for simlab this is 
        li_mean += self.ds.means[mod0] #! 'depth': [0.7302197],
        li_std += self.ds.stds[mod0] #! 'depth': [0.25182092],

        if not 'RGB' in mods[0]:
            img = img[..., None]        # add one dim
        li_img.append(img) #! append mod0 as first channel, add other channels if needed

        # for mod in mods[1:]: #! ignore for initial depth only work
        #     img = self.ds.get_array_A2B(idx=idx, modA=mod, modB=mod0)
        #     li_mean += self.ds.means[mod]
        #     li_std += self.ds.stds[mod]
        #     if 'RGB' != mod:
        #         img = img[...,None]     # add dim
        #     li_img.append(img)

        #! turn list of images with different modalities into one image with channels
        #! not required in our simplified single-depth-modality case
        # img_cb = np.concatenate(li_img, axis=-1)    # last dim, joint mods 

        # augmetation #! image patch is cropped and resized from (512, 424) to (256, 256) only (cropped based on bounding box)
        scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
        img_patch, trans = generate_patch_image(img_cb, bb, do_flip, scale, rot, do_occlusion, input_shape=self.opts.sz_pch[::-1])   # ori to bb, flip first trans later

        #! add an extra dimension to the image to enable use of additional channels (not actually necessary in this case since img is depth map only)
        #! (NOTE ORIGINAL REPLACED WITH SIMPLIFIED EQUIVALENT)
        img_patch = img_patch[..., None] #! img_patch.shape = (256, 256, 1), now applicable for pytorch
        '''
        if img_patch.ndim<3:
            img_channels = 1        # add one channel
            img_patch = img_patch[..., None]
        else:
            img_channels = img_patch.shape[2]   # the channels
        '''

        #! clip the values to 0-255 after scaling (above)
        #! because scaling factor is [1.0, 1.0, 1.0], this operation does nothing
        '''
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)
        '''
            
        # 4. generate patch joint ground truth #! scale ground truth joints to match scaled image patch
        # flip joints and apply Affine Transform on joints
        joints_pch = joints_ori.copy()      # j
        for i in range(len(joints_pch)):  #  jt trans
            joints_pch[i, 0:2] = trans_point2d(joints_pch[i, 0:2], trans)
        stride = sz_pch[0]/out_shp[1]  # jt shrink
        joints_hm = joints_pch/stride #! joints_hm is just 2D joint ground truths scaled to the image patch

        joints_vis = np.ones(n_jt)      # n x 1
        for i in range(len(joints_pch)):        # only check 2d here
            # joints_ori [i, 2] = (joints_ori [i, 2] + 1.0) / 2.  # 0~1 normalize
            joints_vis[i] *= (
                    (joints_pch[i, 0] >= 0) & \
                    (joints_pch[i, 0] < self.opts.sz_pch[0]) & \
                    (joints_pch[i, 1] >= 0) & \
                    (joints_pch[i, 1] < self.opts.sz_pch[1])
            )  # nice filtering  all in range visibile

        hms, jt_wt = generate_target(joints_hm, joints_vis, sz_hm=out_shp[::-1])  # n_gt x H XW
        idx_t, idx_h = ut.nameToIdx(('Thorax', 'Head'), ds.joints_name)
        l_std_hm = np.linalg.norm(joints_hm[idx_h] - joints_hm[idx_t])
        l_std_ori = np.linalg.norm(joints_ori[idx_h] - joints_ori[idx_t])

        # if_vis = False
        # if if_vis:
        #     print('saving feeder data out to rstT')
        #     tmpimg = img_patch.copy().astype(np.uint8)  # rgb
        #     tmpkps = np.ones((n_jt, 3))
        #     tmpkps[:, :2] = joints_pch[:, :2]
        #     tmpkps[:, 2] = joints_vis
        #     tmpimg = vis_keypoints(tmpimg, tmpkps, ds.skels_idx)  # rgb
        #     cv2.imwrite(path.join('rstT', str(idx) + '_pch.jpg'), tmpimg)
        #     hmImg = hms.sum(axis=0)  #
        #     # hm_nmd = hmImg.copy()
        #     # cv2.normalize(hmImg, hm_nmd, beta=255)        # normalize not working
        #     hm_nmd = ut.normImg(hmImg)
        #     # print('hms shape', hms.shape)
        #     cv2.imwrite(path.join('rstT', str(idx) + '_hm.jpg'), hm_nmd)
        #     tmpimg = ut.normImg(tmpimg)
        #     img_cb = vis.hconcat_resize([tmpimg, hm_nmd])
        #     cv2.imwrite(path.join('rstT', str(idx) + '_cb.jpg'), img_cb)

        #! apply transforms to produce final image (toTensor and normalise)
        trans_tch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=li_mean, std=li_std)
            ])
        pch_tch = trans_tch(img_patch) #! imagePatch_Pytorch

        hms_tch = torch.from_numpy(hms) #! heatmaps_Pytorch
        rst = {
            'pch':pch_tch,
            'hms': hms_tch,
            'joints_vis': jt_wt,
            'joints_pch': joints_pch.astype(np.float32),       # in case direct reg
            'l_std_hm':l_std_hm.astype(np.float32),
            'l_std_ori':l_std_ori.astype(np.float32),
            'joints_ori': joints_ori.astype(np.float32),
            'bb': bb.astype(np.float32)     # for recover
        }
        return rst

# define the getData func according to prep  methods for different  jobs
# dct_func_getData = {
#             'SLP_A2J': self.SLP_A2J,
#             'MPPE3D': self.MPPE3D,
#             'jt_hm': self.jt_hm
#         }

# # parser.add_argument('--prep', default='SLP_A2J', help='data preparation method') #! somehow this argument is overwritten to "jt_hm"
# self.func_getData = dct_func_getData[opts.prep] # preparastioin

# def __getitem__(self, index):
#         # call the specific processing
#         rst = self.func_getData(index)
#         return rst



if __name__ == '__main__':
    main()