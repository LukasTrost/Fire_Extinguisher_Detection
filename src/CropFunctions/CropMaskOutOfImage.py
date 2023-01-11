import cv2
import numpy as np
from matplotlib import pyplot as plt
def CropMaskOutOfImage(image, mask):
    mask_positions = []

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            #if mask[row, col, 0] == 255 and mask[row, col, 1] == 255 and mask[row, col, 2] == 255:
            if mask[row, col] == 0:
                mask_positions.append([row, col])

    new_image = np.full([image.shape[0], image.shape[1], 3],255, dtype=np.uint8)

    for mask_pixel in mask_positions:
        new_image[mask_pixel[0], mask_pixel[1]] = image[mask_pixel[0], mask_pixel[1]]

    return new_image
