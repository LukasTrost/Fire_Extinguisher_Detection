from matplotlib import pyplot as plt
import numpy as np
import cv2

def calculate_iou(prediction_mask, ground_truth_mask):
    # Convert masks to binary arrays
    prediction_mask = np.asarray(prediction_mask, dtype=bool)
    ground_truth_mask = np.asarray(ground_truth_mask, dtype=bool)

    # Compute intersection and union masks
    intersection = np.logical_and(prediction_mask, ground_truth_mask)
    union = np.logical_or(prediction_mask, ground_truth_mask)

    # print("intersection")
    # plt.imshow(intersection)
    # plt.show()

    # print("union")
    # plt.imshow(union)
    # plt.show()
    # Calculate areas
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    #print(intersection_area)
    #print(union_area)

    # Calculate IoU
    iou = intersection_area / union_area
    return iou
def CalculateAccuracyAndIoU(mask, trueMask):

    # take 2 binary (black, white) masks and compare their values
    # calculate percentage of differing pixels

    if mask.shape != trueMask.shape:
        print(mask.shape)
        print(trueMask.shape)

        raise ValueError("Mask and trueMask have different shapes")

    #print("prediction")
    #plt.imshow(mask)
    #plt.show()

    #print("groundtruth")
    #plt.imshow(trueMask)
    #plt.show()

    prediction_mask = np.asarray(mask > 0, dtype=np.uint8)
    ground_truth_mask = np.asarray(trueMask > 0, dtype=np.uint8)
    difference_image = cv2.absdiff(prediction_mask, ground_truth_mask)

    iou = calculate_iou(prediction_mask, ground_truth_mask)

    #print(iou)

    #print("difference")
    #plt.imshow(difference_image)
    #plt.show()

    num_differing_pixels = 0

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] != trueMask[row, col]:
                num_differing_pixels += 1

    size = mask.shape[0] * mask.shape[1]

    percentage = (size-num_differing_pixels)/size

    #print(percentage)
    return (percentage), (iou)