from enum import Enum
from dataclasses import dataclass
from typing import Callable
import numpy as np

# TODO %%%%%%%%%%%%%%%%%%% MOVE THESE TO ANOTHER FILE
from D455_testing.comparePreprocessing import set_margin_to_max

class D455_Setups(Enum):
    D455_V1___30_09_24 = 'D455_V1___30_09_24'
    D455_V2___08_10_24 = 'D455_V2___08_10_24'
    D455_V3___08_10_24 = 'D455_V3___08_10_24' 

@dataclass
class D455_Setup:
    setup_name: str # unique name to identify setup (usually the folder name)
    base_path: str # path from repo root to folder containing all images
    crop_heightScale: float
    crop_pShiftDown: float
    crop_pShiftRight: float
    maskObjects: Callable[[np.ndarray], np.ndarray] # take numpy array, apply mask and return numpy array

def getD455SetupProperties(setupEnum: D455_Setups) -> D455_Setup:

    if setupEnum == D455_Setups.D455_V1___30_09_24:
        def applyMask(npImage = np.ndarray) -> np.ndarray:
            return set_margin_to_max(npImage, margin_percent_x=[0.1, 0.155], margin_percent_y=[0.03, 0.])
        
        return D455_Setup(
            setup_name='V01',
            base_path='D455_V01_initialTesting',
            crop_heightScale=0.99, crop_pShiftDown=-0.02, crop_pShiftRight=0.0,
            maskObjects=applyMask
            )
    
    elif setupEnum == D455_Setups.D455_V2___08_10_24:
        return D455_Setup(
            setup_name='V02',
            base_path='D455_V02_singleWithCover',
            crop_heightScale=0.85, crop_pShiftDown=0.03, crop_pShiftRight=0.01,
            maskObjects=applyMask
            )
    
    elif setupEnum == D455_Setups.D455_V3___08_10_24:
        return D455_Setup(
            setup_name='V03',
            base_path='D455_V03_betterSpacing',
            crop_heightScale=0.85, crop_pShiftDown=0.03, crop_pShiftRight=0.01,
            maskObjects=applyMask
            )
    else:
        raise ValueError("Unknown setup")