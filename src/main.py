import cv2
from PIL import Image
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from src.MaskFunctions.DefaultMaskFunction import DefaultMaskFunction
from src.MaskFunctions.Extinguisher_Net import Extinguisher_Net
from src.AdministrativeFunctions.DisplayAndSafe import DisplayAndSave
from src.CropFunctions.CropMaskOutOfImage import CropMaskOutOfImage
# Current implementation Saves Cropped Images in the same parent folder as Original Images


DATA_PATH = "D:/Programmieren/MasterOfDisaster/Experiments_and_Implementations/Extinguisher_Net"
DATA_PATH_ORIGINAL_IMAGES = os.path.join(DATA_PATH, "Resized_Images_640")
DATA_PATH_CROPPED_IMAGES = os.path.join(DATA_PATH, "TestRuns\CroppedImages")

IMAGE_OPEN_TYPES = ["Pillow", "CV2"]
# currently only CV2 supported, might make more if needed
CURRENT_OPEN_TYPE = IMAGE_OPEN_TYPES[1]




if __name__ == '__main__':
    #print(os.listdir(DATA_PATH_ORIGINAL_IMAGES))

    cropfunction = CropMaskOutOfImage
    maskfunctions = [[Extinguisher_Net],["Extinguisher_Net"],
                     [
                          [170,100,100, 180,255,255, 0,100,100, 10,255,255]
                     ]
                     ]
    variablesteps = [
                        [1,1,1, 1,1,1, 1,1,1, 1,1,1]
                    ]



    DisplayAndSave(maskfunctions=maskfunctions, cropfunction=cropfunction, resize=False, resizeDimensions= [30,30],
                   datapath_cropped = DATA_PATH_CROPPED_IMAGES,datapath_original = DATA_PATH_ORIGINAL_IMAGES,
                   displayIdxFromTo = [1,3], displayMasksFromTo = [1,2],
                   maskVariableSteps = variablesteps, displayResults = False, safe_Images_as_well = False,
                   center_Image = True)
