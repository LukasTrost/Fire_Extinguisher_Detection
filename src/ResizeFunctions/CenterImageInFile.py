import cv2
from matplotlib import pyplot as plt

def CenterImageInFile(image):
    # Find the first and last appearance of a completely white pixel in each row and column
    row_first_white = -1
    row_last_white = -1
    col_first_white = -1
    col_last_white = -1
    for row in range(image.shape[0]):
        if row_first_white == -1:
            for col in range(image.shape[1]):
                if image[row, col, 0] == 255 and image[row, col, 1] == 255 and image[row, col, 2] == 255:
                    row_first_white = row
                    break
        else:
            break
    for row in range(image.shape[0] - 1, -1, -1):
        if row_last_white == -1:
            for col in range(image.shape[1]):
                if image[row, col, 0] == 255 and image[row, col, 1] == 255 and image[row, col, 2] == 255:
                    row_last_white = row
                    break
        else:
            break
    for col in range(image.shape[1]):
        if col_first_white == -1:
            for row in range(image.shape[0]):
                if image[row, col, 0] == 255 and image[row, col, 1] == 255 and image[row, col, 2] == 255:
                    col_first_white = col
                    break
        else:
            break
    for col in range(image.shape[1] - 1, -1, -1):
        if col_last_white == -1:
            for row in range(image.shape[0]):
                if image[row, col, 0] == 255 and image[row, col, 1] == 255 and image[row, col, 2] == 255:
                    col_last_white = col
                    break
        else:
            break

    # Crop the image
    cropped_image = image[row_first_white:row_last_white, col_first_white:col_last_white]
    return cropped_image