'''
2d pose estimation handling
'''

from data.SLP_RD import SLP_RD
from data.SLP_FD import SLP_FD
import utils.vis as vis
import utils.utils as ut
import numpy as np
import opt
import cv2
import torch
import json
from os import path
import os
from utils.logger import Colorlogger
from utils.utils_tch import get_model_summary
from core.loss import JointsMSELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
from utils.utils_ds import accuracy, flip_back
from utils.visualizer import Visualizer

'''
BASIC PLAN:

#! (Inside of the runInference Script, basic model setup)
#? load the model architecture (no weights yet)
model = (get the pose net using given function) #* don't need to worry about this, already implemented and I don't need to change anything
#? load the model weights
model.load_state_dict(checkpoint['state_dict'])

#! (Inside of validate.py, prepare the model for evaluation)
#? switch the model to evaluation mode
model.eval()
#? run forward propagation (inference) of the model on the image
outputs = model(input) #* input must be in correct format, comes from the custom dataloader so I need to replicate this format
'''


#  test_loader = DataLoader(dataset=SLP_fd_test, batch_size = opts.batch_size // len(opts.trainset),
#                               shuffle=False, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)
def validate(loader,ds_rd, model, criterion, n_iter=-1, logger=None, opts=None, if_svVis=False, visualizer=None):
	'''
	loop through loder, all res, get preds and gts and normled dist.
	With flip test for higher acc.
	for preds, bbs, jts_ori, jts_weigth out, recover preds_ori, dists_nmd, pckh( dist and joints_vis filter, , print, if_sv then save all these

	:param loader: 
	:param ds_rd: the reader, givens the length and flip pairs
	:param model:
	:param criterion:
	:param optimizer:
	:param epoch:
	:param n_iter:
	:param logger:
	:param opts:
	:return:
	'''
	batch_time = ut.AverageMeter()
	losses = ut.AverageMeter()
	acc = ut.AverageMeter()

	# switch to evaluate mode
	model.eval()

	num_samples = ds_rd.n_smpl
	n_jt = ds_rd.joint_num_ori

	# to accum rst
	preds_hm = []
	bbs = []
	li_joints_ori = []
	li_joints_vis = []
	li_l_std_ori = []
	with torch.no_grad(): # ------------------------------------------------------------------------------> required to prevent updates
		end = time.time()
		for i, inp_dct in enumerate(loader):
			# compute output
			input = inp_dct['pch'] #! torch.Size([60, 1, 256, 256])
			target = inp_dct['hms']
			target_weight = inp_dct['joints_vis']
			bb = inp_dct['bb']
			joints_ori = inp_dct['joints_ori']
			l_std_ori = inp_dct['l_std_ori']
			if i>= n_iter and n_iter>0:     # limiting iters
				break
			outputs = model(input) # --------> run inference on input image #! torch.Size([60, 14, 64, 64])
			if isinstance(outputs, list):
				output = outputs[-1]
			else:
				output = outputs
			output_ori = output.clone()     # original output of original image #! torch.Size([60, 14, 64, 64])

			#! ignore everything to do with flipping
			'''
			if opts.if_flipTest:
				input_flipped = input.flip(3).clone()       # flipped input
				outputs_flipped = model(input_flipped)      # flipped output
				if isinstance(outputs_flipped, list):
					output_flipped = outputs_flipped[-1]
				else:
					output_flipped = outputs_flipped
				output_flipped_ori = output_flipped.clone() # hm only head changed? not possible??
				output_flipped = flip_back(output_flipped.cpu().numpy(),
				                           ds_rd.flip_pairs)
				# output_flipped = torch.from_numpy(output_flipped.copy()).cuda() # N x n_jt xh x w tch ######################################## removed to run on cpu only
				output_flipped = torch.from_numpy(output_flipped.copy()).cpu() # N x n_jt xh x w tch

				# feature is not aligned, shift flipped heatmap for higher accuracy
				if_shiftHM = True  # no idea why
				if if_shiftHM:      # check original
					# print('run shift flip')
					output_flipped[:, :, :, 1:] = \
						output_flipped.clone()[:, :, :, 0:-1]

				output = (output + output_flipped) * 0.5
			'''

			# target = target.cuda(non_blocking=True) ################################################################# removed to run on cpu only
			# target_weight = target_weight.cuda(non_blocking=True) ################################################################# removed to run on cpu only
			loss = criterion(output, target, target_weight)

			num_images = input.size(0)
			# measure accuracy and record loss
			losses.update(loss.item(), num_images)
			_, avg_acc, cnt, pred_hm = accuracy(output.cpu().numpy(),
			                                 target.cpu().numpy())
			acc.update(avg_acc, cnt)

			# preds can be furhter refined with subpixel trick, but it is already good enough.
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			# keep rst
			preds_hm.append(pred_hm)        # already numpy, 2D
			bbs.append(bb.numpy())
			li_joints_ori.append(joints_ori.numpy())
			li_joints_vis.append(target_weight.cpu().numpy())
			li_l_std_ori.append(l_std_ori.numpy())

			if if_svVis and 0 == i % opts.svVis_step:
				sv_dir = opts.vis_test_dir  # exp/vis/Human36M
				# batch version
				mod0 = opts.mod_src[0]
				mean = ds_rd.means[mod0]
				std = ds_rd.stds[mod0]
				img_patch_vis = ut.ts2cv2(input[0], mean, std) # to CV BGR
				# img_patch_vis_flipped = ut.ts2cv2(input_flipped[0], mean, std) # to CV BGR #! ignore everything to do with flipping
				# pseudo change
				cm = getattr(cv2,ds_rd.dct_clrMap[mod0])
				img_patch_vis = cv2.applyColorMap(img_patch_vis, cm)
				# img_patch_vis_flipped = cv2.applyColorMap(img_patch_vis_flipped, cm) #! ignore everything to do with flipping

				# original version get img from the ds_rd , different size , plot ing will vary from each other
				# warp preds to ori
				# draw and save  with index.

				idx_test = i * opts.batch_size  # image index
				skels_idx = ds_rd.skels_idx
				# get pred2d_patch
				pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis
				pred2d_patch[:,:2] = pred_hm[0] / opts.out_shp[0] * opts.sz_pch[1]      # only first
				vis.save_2d_skels(img_patch_vis, pred2d_patch, skels_idx, sv_dir, suffix='-'+mod0,
				                  idx=idx_test)  # make sub dir if needed, recover to test set index by indexing.
				# save the hm images. save flip test
				hm_ori = ut.normImg(output_ori[0].cpu().numpy().sum(axis=0))    # rgb one
				# hm_flip = ut.normImg(output_flipped[0].cpu().numpy().sum(axis=0)) #! ignore everything to do with flipping
				# hm_flip_ori = ut.normImg(output_flipped_ori[0].cpu().numpy().sum(axis=0)) #! ignore everything to do with flipping
				# subFd = mod0+'_hmFlip_ori'
				# vis.save_img(hm_flip_ori, sv_dir, idx_test, sub=subFd)

				# combined
				# img_cb = vis.hconcat_resize([img_patch_vis, hm_ori, img_patch_vis_flipped, hm_flip_ori])        # flipped hm
				# subFd = mod0+'_cbFlip'
				# vis.save_img(img_cb, sv_dir, idx_test, sub=subFd)


			if i % opts.print_freq == 0:
				msg = 'Test: [{0}/{1}]\t' \
				      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
				      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
				      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(loader), batch_time=batch_time,
					loss=losses, acc=acc)
				logger.info(msg)

	preds_hm = np.concatenate(preds_hm,axis=0)      # N x n_jt  x 2
	bbs = np.concatenate(bbs, axis=0)
	joints_ori = np.concatenate(li_joints_ori, axis=0)
	joints_vis = np.concatenate(li_joints_vis, axis=0)
	l_std_ori_all = np.concatenate(li_l_std_ori, axis=0)

	preds_ori = ut.warp_coord_to_original(preds_hm, bbs, sz_out=opts.out_shp)
	err_nmd = ut.distNorm(preds_ori,  joints_ori, l_std_ori_all)
	ticks = np.linspace(0,0.5,11)   # 11 ticks
	pck_all = ut.pck(err_nmd, joints_vis, ticks=ticks)

	# save to plain format for easy processing
	rst = {
		'preds_ori':preds_ori.tolist(),
		'joints_ori':joints_ori.tolist(),
		'l_std_ori_all': l_std_ori_all.tolist(),
		'err_nmd': err_nmd.tolist(),
		'pck': pck_all.tolist()
	}

	return rst