import cv2
from PIL import Image
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from src.MaskFunctions.DefaultMaskFunction import DefaultMaskFunction
from src.AdministrativeFunctions.DisplayAndSafe import DisplayAndSave
from src.CropFunctions.CropMaskOutOfImage import CropMaskOutOfImage
# Current implementation Saves Cropped Images in the same parent folder as Original Images
DATA_PATH = "D:\Programmieren\MasterOfDisaster\Implementierungen\TestImages_and_Results"
DATA_PATH_ORIGINAL_IMAGES = os.path.join(DATA_PATH, "OriginalImages_Resized")
DATA_PATH_CROPPED_IMAGES = os.path.join(DATA_PATH, "TestRuns\CroppedImages")

IMAGE_OPEN_TYPES = ["Pillow", "CV2"]
# currently only CV2 supported, might make more if needed
CURRENT_OPEN_TYPE = IMAGE_OPEN_TYPES[1]

# TODO might wanna check out https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/
# or https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/

#TODO stelle schon mal ein einfaches opening closing dar als erste methode
# stelle nur die maske des watersheds dar in einem weissen bild und guck ob du damit was amchen kannst
#   - schreibe funktion das anhand einer maske ein bild ausschneidet
#   recherche was ich noch so versuchen könnte
#   interessante Idee:
#   wenn Kanten gut funktioniert, / mehrere Objekt auf einem Bild, macht es sinn nur das größte masken objekt zu nehmen

# TODO falls Lösung Algorithmus der wie bei KG durch alle möglichen variablen geht, z.b. Größe des Strukturelementes variieren oder anzahl der closing iterationen


# TODO überprüfe ob überall auch binarisiert und nicht nur grau (np.where)

if __name__ == '__main__':
    #print(os.listdir(DATA_PATH_ORIGINAL_IMAGES))

    cropfunction = CropMaskOutOfImage
    maskfunctions = [[DefaultMaskFunction,DefaultMaskFunction],["Defaultname1","Defaultname2"],
                     [  [[0,10],[10,30]],[[0,10],[5]]  ]     ]
    variablesteps = [[2,1],[1,1],[2]]

    # display parameters are in realvalues not array values, so to display the first 3 images type displayIdxFromTo = [1,3]

    DisplayAndSave(maskfunctions=maskfunctions, cropfunction=cropfunction, resize=True, resizeDimensions= [30,30],
                   datapath_cropped = DATA_PATH_CROPPED_IMAGES,datapath_original = DATA_PATH_ORIGINAL_IMAGES,
                   displayIdxFromTo = [1,3], displayMasksFromTo = [1,2], displayMaskVariationsFromTo = [1,2],
                   maskVariableSteps = variablesteps)
