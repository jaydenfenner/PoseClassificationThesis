from enum import Enum

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

class constants:
    ''' Constants required when preprocessing images'''
    model_input_size = [256, 256] #! required image size to be fed into the model
    # model_output_size = [64, 64] #! output heatmap size of the model
    
    mean_depth = 0.7302197 #! mean and std used for normalisation of depth image
    std_depth = 0.25182092

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
    skels_idx = nameToIdx(skels_name, joints_name=joints_name)