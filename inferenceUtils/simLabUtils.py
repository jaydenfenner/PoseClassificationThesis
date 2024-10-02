import scipy.io as sio
import cv2
import numpy as np
import os

def getSimLabDepthGTsForSubject(subj):
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