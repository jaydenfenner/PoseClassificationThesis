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
    preprocessingConstants = {}
    # mean and std used for normalisation of depth image
    mean_depth = 0.7302197 
    std_depth = 0.25182092

    '''
    - simLab cropped(0-255):
        - bed:  min: 180,  peak 190,  max: 208
        - floor:  min: 237,  peak 242,  max: 246
        - floor-bed peak diff = 52  (21.5% of floor peak)  (27% of bed peak)
        - body-bed range = 28
    - danaLab cropped (0-255):
        - bed:  min: 162,  peak 179,  max: 186 --------> min_on_side: 152
        - floor:  min: 212,  peak 220,  max: 227
        - floor-bed peak diff = 41  (18.6% of floor peak)  (23% of bed peak)
        - body-bed range = 24
    - D455_V3 cropped (0-255):
        - bed:  min: 204,  peak 222,  max: 232
    '''
    # bed/subject min and max 0-255 values in danaLab (training data)
    # for pose flat on back (used for scaling D455 images to match training data)
    danaLab_minBedDepth = 162
    danaLab_maxBedDepth = 186

    # Joint names, indexes and number, taken from dataset reader
    numberOfJoints = 14 #? Required to instantiate model (out channels)
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
        "L_Elbow", "L_Wrist", "Thorax", "Head", 
        "Pelvis", "Torso","Neck") # NOTE NO PELVIS, Torso, or Neck
    skels_idx = nameToIdx(skels_name, joints_name=joints_name)