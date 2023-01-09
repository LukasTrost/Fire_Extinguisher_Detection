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

def DisplayAndSave(cropfunction =CropMaskOutOfImage, maskfunction = DefaultMaskFunction, resize = True,
                   resizeDimensions = [480,640], datapath_original = "", datapath_cropped = "", datapath_mask = "",
                   displayAmount = 3):
    overallFolder = os.listdir(datapath_original)
    imageObject = []

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
                print(os.path.join(path,itemName))

        if image is not None:
            print("Image was found so procede")
            if resize:
                image = ResizeFunction(image, resizeDimensions)
            mask = maskfunction(image)
            croppedImage = CropMaskOutOfImage(image)
            centeredImage = CenterImageInFile(croppedImage)

            if trueMask is not None:
                accuracy = CalculateAccuracy(mask, trueMask)
            else:
                print("No Mask Found")
            imageObject.append([image, croppedImage, itemName.title(), mask, centeredImage, accuracy, folder])

        else:
            print("ERROR,no Image in folder: ")
    image_and_crop = [subarray[:2] for subarray in imageObject]
    DisplayImagesAndCroppedImages(image_and_crop, 3)
    CreateFolderForResults(imageObject,datapath_cropped)
