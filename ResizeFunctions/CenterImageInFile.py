import cv2
from matplotlib import pyplot as plt

def CenterImageInFile(image):

    # Load the image
    #image = cv2.imread('image.jpg')

    # Find all non-zero points
    points = cv2.findNonZero(image)

    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(points)

    # Crop the image
    cropped_image = cv2.getRectSubPix(image, (w, h), (x, y))

    plt.imshow(cropped_image)
    return 69