from src.ResizeFunctions.ResizeImage import ResizeFunction
from src.DisplayFunctions.DisplayNormalAndCropped import DisplayImagesAndCroppedImages
from src.FolderFunctions.CreateFolderHierarchy import CreateFolderForResults
import os
import cv2
from src.MaskFunctions.DefaultMaskFunction import DefaultMaskFunction
from src.CropFunctions.CropMaskOutOfImage import CropMaskOutOfImage
from src.ResizeFunctions.CenterImageInFile import CenterImageInFile
from src.AccuracyCalculation.CalculateAccuracy import CalculateAccuracy
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def DisplayAndSave(cropfunction =CropMaskOutOfImage, maskfunction = DefaultMaskFunction, resize = True,
                   resizeDimensions = [480,640], datapath_original = "", datapath_cropped = "", datapath_mask = "",
                   displayAmount = 3):
    overallFolder = os.listdir(datapath_original)
    imageObject = []
    pbar = tqdm(total=len(overallFolder), colour='#FF69B4')
    for folder in overallFolder:
        image = None
        trueMask = None
        accuracy = "No Ground Truth"
        path =os.path.join(datapath_original,folder)
        datasetFolder = os.listdir(path)
        print(datasetFolder)
        for itemName in datasetFolder:
            if "mask" in itemName:
                trueMask = cv2.imread(os.path.join(path,itemName))
                print(itemName)
            else:
                image = cv2.imread(os.path.join(path,itemName))
                #print(os.path.join(path,itemName))

        if image is not None:
            #print("Image was found so procede")
            if resize:
                image = ResizeFunction(image, resizeDimensions)
            mask = maskfunction(image)
            croppedImage = CropMaskOutOfImage(image, mask)
            centeredImage = CenterImageInFile(croppedImage)


            if trueMask is not None:
                if resize:
                    trueMask = ResizeFunction(trueMask, resizeDimensions)
                accuracy = CalculateAccuracy(np.where(mask > 0, 1, 0), np.where(cv2.cvtColor(trueMask, cv2.COLOR_BGR2GRAY) > 0, 1, 0))
            else:
                print("No Mask Found")
            imageObject.append([image, croppedImage, folder.title(), mask, centeredImage, accuracy, folder])
            pbar.update(1)
        else:
            print("ERROR,no Image in folder: ")
    image_and_crop = [subarray[:2] for subarray in imageObject]
    DisplayImagesAndCroppedImages(image_and_crop, 3)
    CreateFolderForResults(imageObject,datapath_cropped)
