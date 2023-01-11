from matplotlib import pyplot as plt
def CalculateAccuracy(mask, trueMask):

    # take 2 binary (black, white) masks and compare their values
    # calculate percentage of differing pixels

    if mask.shape != trueMask.shape:
        print(mask.shape)
        print(trueMask.shape)

        raise ValueError("Mask and trueMask have different shapes")

    num_differing_pixels = 0

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] != trueMask[row, col]:
                num_differing_pixels += 1

    size = mask.shape[0] * mask.shape[1]

    percentage = (size-num_differing_pixels)/size

    #print(percentage)
    return(percentage)