from matplotlib import pyplot as plt
import cv2

def DisplayImagesAndCroppedImages(images_and_crops, displayAmount):

    #print(len(images_and_crops))
    amountToDisplay = min([len(images_and_crops),displayAmount])
    #print(amountToDisplay)
    fig, ax = plt.subplots(1, amountToDisplay*2)
    # Print the contents
    for pair_idx in range(0,amountToDisplay,1):
        #print("a")
        #plt.imshow(images_and_crops[pair_idx][0])
        ax[pair_idx*2].imshow(cv2.cvtColor(images_and_crops[pair_idx][0],cv2.COLOR_BGR2RGB))
        ax[pair_idx*2+1].imshow(cv2.cvtColor(images_and_crops[pair_idx][1],cv2.COLOR_BGR2RGB))
        ax[pair_idx*2].axis('off')
        ax[pair_idx*2+1].axis('off')
    plt.show()