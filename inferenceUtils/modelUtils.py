import torch
from inferenceUtils.constants import PretrainedModels, constants, getModelProperties
import importlib
import os

# Function to dynamically load the model and get the pose network
def _load_pose_net(model: PretrainedModels):
    '''
    Load the appropriate model architecture from SLP repo models/model_name/get_pose_net
    '''
    try:
        modelProperties = getModelProperties(model)
        # Import the module dynamically from the 'models' package
        model_module = importlib.import_module(f'model.{modelProperties.model_name}')
        # Call the 'get_pose_net' function from the imported module
        pose_net = model_module.get_pose_net(
            in_ch = modelProperties.in_channels, 
            out_ch = constants.numberOfJoints
            )
        return pose_net
    
    except ModuleNotFoundError:
        raise ValueError(f"Model '{modelProperties.model_name}' not found.")
    except AttributeError:
        raise ValueError(f"Model '{modelProperties.model_name}' does not have a 'get_pose_net' function.")

def loadPretrainedModel(modelType: PretrainedModels):
    '''
    Load the relevant pretrained model for inference on cpu:
    - load the model architecture 
        - (using {modelName}.get_pose_net from SLP repo)
    - load the pretrained weights
        - (from pretrainedModels/{ pretrainedModelName }/model_dump/checkpoint.pth)
    - NOTE hardcoded to use cpu
        - (not sure how to revert this)
    '''
    # get model (architecture only)
    model = _load_pose_net(modelType) # 1 channel (depth), 14 joints
    pretrained_model_name = getModelProperties(modelType=modelType).pretrained_model_name
    pretrained_model_path = os.path.join("pretrainedModels", pretrained_model_name, 'model_dump/checkpoint.pth')
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu')) #! note extra argument to use cpu
    model.load_state_dict(checkpoint['state_dict'])

    # model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids) # paralellise the torch operations #! removed since I'm using cpu only
    model = model.to('cpu') # send model to cpu

    return model