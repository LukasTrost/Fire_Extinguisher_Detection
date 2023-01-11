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

def DisplayAndSave(cropfunction =CropMaskOutOfImage, maskfunctions = [DefaultMaskFunction], resize = True,
                   resizeDimensions = [480,640], datapath_original = "", datapath_cropped = "",
                   displayIdxFromTo = [1,1], displayMasksFromTo = [1,1], maskVariableSteps = [[2],[2,2]],
                   displayResults = False, safe_Images_as_well = False, center_Image = False):
    overallFolder = os.listdir(datapath_original)
    print(overallFolder)
    imageObject = []
    pbar = tqdm(total=len(overallFolder), colour='#FF69B4')
    for folder in overallFolder:
        image = None
        trueMask = None
        path =os.path.join(datapath_original,folder)
        datasetFolder = os.listdir(path)
        #print(datasetFolder)
        for itemName in datasetFolder:
            if "mask" in itemName:
                trueMask = cv2.imread(os.path.join(path,itemName))
                #print(itemName)
            else:
                image = cv2.imread(os.path.join(path,itemName))
                #print(itemName)
                #print(os.path.join(path,itemName))

        if image is not None:
            #print("Image was found so procede")
            if resize:
                image = ResizeFunction(image, resizeDimensions)

            if trueMask is not None:
                if resize:
                    trueMask = ResizeFunction(trueMask, resizeDimensions)
            else:
                print("No Ground Truth Mask Found")


            masks = []
            croppedImages = []
            centeredImages = []
            accuracies = []
            concrete_value_combinations = []

            for idx, maskfunction in enumerate (maskfunctions[0]):
                concrete_values, mask_variations = maskfunction(image, maskVariableSteps[idx], maskfunctions[2][idx])

                masks.append(mask_variations)
                croppedImage_variations = []
                centeredImage_variations = []
                accuracy_variations = []

                for variation in mask_variations:
                    if safe_Images_as_well or displayResults:
                        croppedImage_variation = cropfunction(image, variation)
                        croppedImage_variations.append(croppedImage_variation)

                    if safe_Images_as_well and center_Image:
                        centeredImage_variation = CenterImageInFile(croppedImage_variation)
                        centeredImage_variations.append(centeredImage_variation)

                    if trueMask is not None:
                        accuracy_variation = CalculateAccuracy(variation, np.where(cv2.cvtColor(trueMask, cv2.COLOR_BGR2GRAY) > 0, 1, 0))
                        accuracy_variations.append(accuracy_variation)
                    else:
                        accuracy_variations.append("No Ground Truth")

                croppedImages.append(croppedImage_variations)
                centeredImages.append(centeredImage_variations)
                accuracies.append(accuracy_variations)
                concrete_value_combinations.append(concrete_values)
            imageObject.append([image, croppedImages, folder.title(), masks, centeredImages, accuracies, folder, maskfunctions[1], concrete_value_combinations])
            pbar.update(1)
        else:
            print("ERROR,no Image in folder: ")
    #TODO replace with list comprehension of the sort displayObject = [subarray[:2] for subarray in imageObject]

    if displayResults:
        displayObject = []
        for imageIdx in range(displayIdxFromTo[0]-1, min([displayIdxFromTo[1], len(imageObject)]),1):

            maskObject = []
            croppedObject = []
            concrete_values_Object = []

            for maskIDx in range(displayMasksFromTo[0]-1, min([displayMasksFromTo[1], len(maskfunctions)]), 1):
                maskObject.append(imageObject[imageIdx][7][maskIDx])
                all_concrete_values = []
                all_cropped_images = []
                for variationIdx in range (0, len(imageObject[imageIdx][8][maskIDx]), 1):
                    all_cropped_images.append(imageObject[imageIdx][1][maskIDx][variationIdx])
                    all_concrete_values.append(imageObject[imageIdx][8][maskIDx][variationIdx])

                croppedObject.append(all_cropped_images)
                concrete_values_Object.append(all_concrete_values)

            #print(imageObject[imageIdx][0:2])
            displayObject.append([imageObject[imageIdx][0],croppedObject,maskObject,concrete_values_Object])

        DisplayImagesAndCroppedImages(displayObject)

    CreateFolderForResults(imageObject,datapath_cropped, safe_Images_as_well, center_Image)
