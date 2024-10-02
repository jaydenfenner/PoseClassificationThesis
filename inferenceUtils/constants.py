from enum import Enum
from dataclasses import dataclass
from typing import List

#! stolen from utils.utils
def nameToIdx(name_tuple, joints_name):  # test, tp,
    '''
    from reference joints_name, change current name list into index form
    :param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
    :param joints_name:
    :return:
    '''
    jtNm = joints_name
    if type(name_tuple[0]) == tuple:
        # Transer name_tuple to idx
        return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
    else:
        # direct transfer
        return tuple(jtNm.index(tpl) for tpl in name_tuple)
    
class CoverType(Enum):
        COVER1 = 'cover1'
        COVER2 = 'cover2'
        UNCOVER = 'uncover'

class PretrainedModels(Enum):
    HRPOSE_DEPTH = 'SLP_depth_u12_HRpose_exp'
      
@dataclass
class ModelProperties:
    model_name: str # model folder name in SLP repo
    model_input_size: List[int] # model folder name in SLP repo
    model_output_size: List[int] # model folder name in SLP repo
    pretrained_model_name: str # fileName for the pretrained model checkpoint file
    in_channels: int # input channels, depends on modalities

def getModelProperties(modelType: PretrainedModels):
    if modelType == PretrainedModels.HRPOSE_DEPTH:
        return ModelProperties(
            model_name='HRpose',
            model_input_size=[256, 256],
            model_output_size=[64, 64],
            pretrained_model_name = "SLP_depth_u12_HRpose_exp",
            in_channels = 1
            )
    else:
        raise ValueError("Unknown model")

class constants:
    ''' Constants required when preprocessing images'''
    mean_depth = 0.7302197 #! mean and std used for normalisation of depth image
    std_depth = 0.25182092

    #! required to instantiate model
    numberOfJoints = 14

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
    skels_idx = nameToIdx(skels_name, joints_name=joints_name)