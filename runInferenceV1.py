'''
Copy of main.py stripped to help creating inference script

python main.py --modelConf config/[modelName].conf --if_test

IMPORTANT NOTES
- note the check for depth in the arguments and mention of using bounding box, this might impact inference
    - (need to check what opts.if_bb actually does)
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
# from validateCustom import validate
from main import validate

'''
############################################################################################################
# best model saved to:
torch.save(ckp, os.path.join(opts.model_dir, 'model_best.pth'))

best_model_saved_path =
    opts.model_dir + '/model_best.pth' # opts.model_dir = osp.join(opts.exp_dir, 'model_dump')
    opts.exp_dir + '/model_dump' + '/model_best.pth' # opts.exp_dir = osp.join(opts.output_dir, nmT)
    opts.output_dir + nmT + '/model_dump' + '/model_best.pth' # parser.add_argument('--output_dir', default='output')
    'output' + nmT + '/model_dump' + '/model_best.pth' 

    # WHERE nmT IS THE NAME OF THE TEST/EXPERIMENT, generated from the options, etc used
    nmT = '-'.join(opts.trainset)
    nmT = '_'.join([nmT, modStr, covStr, suffix_train, opts.suffix_exp_train])  # ds+ ptn_suffix+ exp_suffix


############################################################################################################
########## when loading the sate from a prior experiment:
exec('from model.{} import get_pose_net'.format(opts.model))  # pose net in

model = get_pose_net(in_ch=opts.input_nc, out_ch=n_jt) # get model

checkpoint_file = os.path.join(opts.model_dir, 'checkpoint.pth')

checkpoint = torch.load(checkpoint_file)
        # begin_epoch = checkpoint['epoch']
        # best_perf = checkpoint['perf']
        # last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])  # here should be cuda setting
        # losses = checkpoint['losses']
        # accs = checkpoint['accs']

############################################################################################################
##### when training the model (EDITED FOR CPU)
model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids).cpu()




# evaluate on validation set    to update
            rst_test = validate(
                test_loader, SLP_rd_test, model, criterion,
                n_iter=n_iter, logger=logger, opts=opts)   # save preds, gt, preds_in ori, idst_normed to recovery, error here for last epoch?



# single test with loaded model, save the result
rst_test = validate(
    test_loader, SLP_rd_test, model, criterion,
    n_iter=n_iter, logger=logger, opts=opts, if_svVis=True)  # save preds, gt, preds_in ori, idst_normed to recovery
pck_all = rst_test['pck']

# perf_indicator = pck_all[-1][-1]  # last entry of list
pckh05 = np.array(pck_all)[:, -1]        # why only 11 pck??
titles_c = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
ut.prt_rst([pckh05], titles_c, ['pckh0.5'], fn_prt=logger.info)

# RESULTS JSON LOCATION: 'outputs/nmT/result/' + opts.nmTest + '.json'
pth_rst = path.join(opts.rst_dir, opts.nmTest + '.json')
with open(pth_rst, 'w') as f:
    json.dump(rst_test, f)

############################################################################################################
'''


# arguments?
opts = opt.parseArgs()
# print(f"runInference 101 opts.prep: {opts.prep}")
######################################################## MY ARGUMENTS
pretrained_model_name = "SLP_depth_u12_HRpose_exp" # name of the pretrained model folder

# parser.add_argument('--SLP_set', default='danaLab', help='[danaLab|simLab] for SLP section')
opts.SLP_set = 'simLab'

# parser.add_argument('--mod_src', nargs='+', default=['IR'],
#     help='source modality list, can accept multiple modalities typical model [RGB|IR|depthRaw| PMarray]')
#! NOTE TEMPORARY SWAP TO DEPTHRAW
opts.mod_src = ['depthRaw']
# opts.mod_src = ['depth']

# parser.add_argument('--cov_li', nargs='+', default=['uncover', 'cover1', 'cover2'], help='the cover conditions')
opts.cov_li = ['cover2']
########################################################
if 'depth' in opts.mod_src[0]:  # the leading modalities, only depth use tight bb other raw image size
    opts.if_bb = True  # not using bb, give ori directly
else:
    opts.if_bb = False  #
exec('from model.{} import get_pose_net'.format(opts.model))  # pose net in
opts = opt.aug_opts(opts)  # add necesary opts parameters   to it

#! needs to be added after the aug
# parser.add_argument('--yn_flipTest', default='y')
# yn_dict = {'y': True, 'n': False}
# opts.if_flipTest = yn_dict[opts.yn_flipTest]
opts.if_flipTest = False


############################################################################################################
# run inference (work in progress)
############################################################################################################
def main():
    # logger setup (can't hurt surely)
    if_test = opts.if_test
    if if_test:
        log_suffix = 'test'
    else:
        log_suffix = 'train'
    logger = Colorlogger(opts.log_dir, '{}_logs.txt'.format(log_suffix))    # avoid overwritting, will append
    opt.set_env(opts)
    opt.print_options(opts, if_sv=True)

    n_jt = SLP_RD.joint_num_ori

    # get model (architecture only)
    model = get_pose_net(in_ch=opts.input_nc, out_ch=n_jt) # prev opts.input_nc

    # load in pretrained weights and initialise model ###########################################################################
    pretrained_model_path = os.path.join("pretrainedModels", pretrained_model_name, 'model_dump/checkpoint.pth')
    logger.info("=> loading checkpoint '{}'".format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu')) ######################### note extra argument to use cpu
    model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids).cpu() # paralellise the torch operations

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(      # try to not use weights
        use_target_weight=True
    ).cuda()


    # feeder and reader for the test set
    SLP_rd_test = SLP_RD(opts, phase=opts.test_par)  # all test result      # can test against all controled in opt
    SLP_fd_test = SLP_FD(SLP_rd_test,  opts, phase='test', if_sq_bb=True)
    test_loader = DataLoader(dataset=SLP_fd_test, batch_size = opts.batch_size // len(opts.trainset),
                              shuffle=False, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)


    n_iter = opts.trainIter  # only for test purpose     quick test

    # single test with loaded model, save the result
    logger.info('----run final test----')
    
    rst_test = validate(
        test_loader, SLP_rd_test, model, criterion,
        n_iter=n_iter, logger=logger, opts=opts, if_svVis=True)  # save preds, gt, preds_in ori, idst_normed to recovery
    pck_all = rst_test['pck']

    # perf_indicator = pck_all[-1][-1]  # last entry of list
    pckh05 = np.array(pck_all)[:, -1]        # why only 11 pck??
    titles_c = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
    ut.prt_rst([pckh05], titles_c, ['pckh0.5'], fn_prt=logger.info)
    pth_rst = path.join(opts.rst_dir, opts.nmTest + '.json')
    with open(pth_rst, 'w') as f:
        json.dump(rst_test, f)
    

if __name__ == '__main__':
    main()
