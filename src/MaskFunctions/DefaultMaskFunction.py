import cv2
import numpy as np
def DefaultMaskFunction(image):
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask > 0, 1, 0)
    return mask