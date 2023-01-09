import cv2
def ResizeFunction(image, resizeDimensions):
    return cv2.resize(image, (resizeDimensions[0], resizeDimensions[1]))