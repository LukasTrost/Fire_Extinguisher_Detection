import cv2
from PIL import Image
import os
from matplotlib import pyplot as plt
from MaskFunctions.DefaultMaskFunction import DefaultMaskFunction
from AdministrativeFunctions.DisplayAndSafe import DisplayAndSave
from CropFunctions.CropMaskOutOfImage import CropMaskOutOfImage
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

if __name__ == '__main__':
    print("hi")
    print(os.listdir(DATA_PATH_ORIGINAL_IMAGES))

    cropfunction = CropMaskOutOfImage
    maskfunction = DefaultMaskFunction
    DisplayAndSave(maskfunction=maskfunction, cropfunction=cropfunction, resize=True, resizeDimensions= [480,640],
                   datapath_cropped = DATA_PATH_CROPPED_IMAGES,datapath_original = DATA_PATH_ORIGINAL_IMAGES,
                   displayAmount = 2)

    # TODO Progress bar und debugger
