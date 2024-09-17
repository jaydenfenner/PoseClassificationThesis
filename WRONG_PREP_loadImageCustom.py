'''
GOAL OF THIS FILE:

The goal of this file is to replicate the loading of a single depth image from the 
#? simLab/00001/depth/uncover/image_000001.png
image in a suitable format for inference
'''

import numpy as np
import cv2

def main():
    img = readRawImage(1)
    img = performRequiredAugs(img)

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

# TODO this is my own custom function for use in preprocessing images
class myDefinedConstants:
    df_cut=0.03
    d_bed = 2.264 #! taken from simLab test set in SLP_RD
    d_cut = d_bed - df_cut

    std = 0.54 #! defined standard deviation for depth image, need to normalise to match this
    mean = d_cut + 0.5 #! mean for depth image, defined based on set depth above cutoff, need to normalise to match this
    bg = d_cut + 0.75  #! background depth defined as constant for all cutoff areas

def readRawImage(idx:int, cover='u'):
    '''
    Take an SLP image index, find the file and read the raw depth image as a numpy array
    '''
    coverCondition = 'cover1' if (cover == '1') else 'cover2' if (cover == '2') else 'uncover'
    idxWithZeros = "{:05d}".format(idx)
    path_to_rawDepth = f'SLP/simLab/{idxWithZeros}/depthRaw/{coverCondition}/0{idxWithZeros}.npy' #* note extra zero on npy file
    img = np.load(path_to_rawDepth)
    print(img)
    print(f"shape: {img.shape}")
    return img

''''''
def performRequiredAugs(img):
    depthR = img # img comes in as raw depth numpy array
    depthM = depthR / 1000. #? divide everything by 1000 (for some reason)
    depthM[depthM > myDefinedConstants.d_cut] = myDefinedConstants.bg #* trim everything below the bed (past cutoff) and set to predefined bg value
    depth = (depthM - myDefinedConstants.mean) / myDefinedConstants.std #* normalise to set mean and standard dev

    #! taken from SLP_RD:
    n_frm = 45  # fixed #? There are 45 positions for each subject, captured with 3 different cover conditions while the subj stayed still
    if True: # if 'simLab' in dsFd: #* just setting since always true
        n_subj = 7
        d_bed = 2.264  # as meter
        n_split = 0  # the split point of train and test (sim for test only)
    idxs_subj = range(n_subj)
    pthDesc_li = []
    for i in idxs_subj:
            for cov in ['uncover']: #opts.cov_li:  # add pth Descriptor #* just setting hardcoded for this
                for j in range(n_frm):
                    pthDesc_li.append([i + 1, cov, j + 1])  # file idx 1 based,  sample idx 0 based
    n_smpl = len(pthDesc_li)
    
    idx_smpl = 1 #! as an example just take the first image
    id_subj, cov, id_frm = pthDesc_li[idx_smpl]    # id for file , base 1

    # joints_gt_depth_t = np.array(list( #! basically just shift the RGB ground truths for the depth camera since they are slightly offset
	# 	map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB2depth)[0], joints_gt_RGB_t[:, :, :2])))

    #! get_bbox(rt_xy=1) just returns a SQUARE bounding box centred on the true box, with smaller dim (w/h) scaled up to match the larger, margin is 1.2x side lengths
    #! box given as: [width_with_margin, height_with_margin, (min x coord of box), (min y coord of box)] note box is square and has 1.2x side lengths
    li_bb_sq_depth = []
    li_bb_sq_depth.append(np.array(list(map(lambda x: ut.get_bbox(x, rt_xy=1), joints_gt_depth_t))))
    # bbox = getattr(self, 'li_bb_sq_depth')[id_subj - 1][id_frm - 1] #! self.li_bb_sq_depth[ SUBJECT_NUMBER_OUTOF_7 ][ POSE_NUMBER_OUTOF_45 ]

    #! recover centre and save width + height of SQUARE bbox with 1.2x side lengths
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    # scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False #! from original, basically no transforms
    # img_patch, trans = ut.generate_patch_image(depth, bb, do_flip=false, scale=1.0, rot=0.0, do_occlusion=false, sz_std=[288, 288] )
    # trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, sz_std[0], sz_std[1], scale, rot, inv=False)
    #! above with arguments only scales the image from input size of (y,x)512x424 to 288x288 and can be replaced with the below:
    #? crop the image to the bounding box
    #? resize to match the required input resolution
    img = cv2.resize(img, (288, 288))


    print(f"depth shape: {depth.shape}")
    print(depth)
    return depth
''''''

# can define transforms here based on train (aug) or test
def SLP_A2J(self, idx):  # outside to costly
    '''
    Pre func. All major preprocessing will be here. __getItem__ only a wrapper so no operations there. All ds inteface provides final ts format , return dict rst with data required by the specific models.
    from SLP ds get all the model(itop) compabible format for training. There seems a counter clockwise operatoion in A2J
    will transform into the C x H xW format
    right now , there is only patch and jt,  may add vis or if_depth later
    :param rt_xy: the x,y ratio of the bb
    :return:
    '''
    # param
    std = 0.54 #! defined standard deviation for depth image, need to normalise to match this
    # define the fill and cut part
    d_cut = self.d_cut
    mean = d_cut + 0.5 #! mean for depth image, defined based on set depth above cutoff, need to normalise to match this
    bg = d_cut + 0.75  #! background depth defined as constant for all cutoff areas
    # get depth, joints, bb
    depthR, joints_gt, bb = self.ds.get_array_joints(idx, mod='depthRaw', if_sq_bb=self.if_sq_bb)  # raw depth
    depthM = depthR/1000.
    depthM[depthM>d_cut] = bg       # cut everything to bg
    depth = (depthM - mean) / std   # norm then factor it #! depth is processed to cut out anything 

    scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False #! from original, basically no transforms
    img_patch, trans = ut.generate_patch_image(depth, bb, do_flip, scale, rot, do_occlusion, sz_std=[288, 288] )# use the
    jt = joints_gt.copy()
    for i in range(len(joints_gt)):  # 2d first for boneLen calculation
        jt[i, 0:2] = ut.trans_point2d(joints_gt[i, 0:2], trans)  # to pix:patch under input_shape size

    jt[:, 2] = jt[:, 2] * self.fc_depth # to pixel  *50

    # single raw or
    if img_patch.ndim < 3:  # gray or depth
        img_patch = img_patch[None, :, :]   # add channel to C H W
    else:
        img_patch = img_patch.transpose([2, 0, 1]) # RGB to  CHW
    # to tensor
    arr_tch = torch.from_numpy(img_patch.copy())  # the rotate make negative index along dim
    jt_tch = torch.from_numpy(jt.copy())
    bb_tch = torch.from_numpy(bb)
    rst = {'arr_tch':arr_tch, 'jt_tch':jt_tch, 'bb_tch':bb_tch}
    return rst


# define the getData func according to prep  methods for different  jobs
# dct_func_getData = {
#             'SLP_A2J': self.SLP_A2J,
#             'MPPE3D': self.MPPE3D,
#             'jt_hm': self.jt_hm
#         }

# #! parser.add_argument('--prep', default='SLP_A2J', help='data preparation method')
# self.func_getData = dct_func_getData[opts.prep] # preparastioin

# def __getitem__(self, index):
#         # call the specific processing
#         rst = self.func_getData(index)
#         return rst



if __name__ == '__main__':
    main()