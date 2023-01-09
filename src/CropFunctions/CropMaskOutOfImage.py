import cv2
import numpy as np
def CropMaskOutOfImage(image, mask):
    mask_positions = []

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col, 0] == 255 and image[row, col, 1] == 255 and image[row, col, 2] == 255:
                mask_positions.append([row, col])

    new_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    new_image.fill(0)

    for mask_pixel in mask_positions:
        new_image[mask_pixel[0], mask_pixel[1]] = image[mask_pixel[0], mask_pixel[1]]

    return new_image
