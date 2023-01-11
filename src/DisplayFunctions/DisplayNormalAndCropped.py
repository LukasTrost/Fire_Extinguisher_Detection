from matplotlib import pyplot as plt
import cv2

def DisplayImagesAndCroppedImages(images_and_crops):


    print(len(images_and_crops[0]))
    print(len(images_and_crops[0][0]))
    print(len(images_and_crops[0][1]))
    print(len(images_and_crops[0][2]))
    # Print the contents

    for mask_Idx in range(0,len(images_and_crops[0][2]),1):
        name = "" + images_and_crops[0][2][mask_Idx]
        for mask_vari_Idx in range(0,len(images_and_crops[0][3][mask_Idx]),1):
            fig, ax = plt.subplots(1, len(images_and_crops) * 2)
            for img_idx in range(len(images_and_crops)):
                ax[img_idx * 2].imshow(cv2.cvtColor(images_and_crops[img_idx][0], cv2.COLOR_BGR2RGB))
                ax[img_idx * 2 + 1].imshow(cv2.cvtColor(images_and_crops[img_idx][1][mask_Idx][mask_vari_Idx], cv2.COLOR_BGR2RGB))
                ax[img_idx*2].axis('off')
                ax[img_idx*2+1].axis('off')
            word = ""
            for name_part in range (1,len(images_and_crops[img_idx][3][mask_Idx][mask_vari_Idx]),2):
                word = word + str(images_and_crops[0][3][mask_Idx][mask_vari_Idx][name_part])
            fig.suptitle(name + word,fontsize=16)
            plt.show()
"""



    for mask_Idx in range(0,len(images_and_crops[0][2]),1):
        fig, ax = plt.subplots(1, len(images_and_crops) * 2)
        for pair_idx in range(0,len(images_and_crops),1):
            for variationIdx in range(0,len(images_and_crops[]))
            #print("a")
            #plt.imshow(images_and_crops[pair_idx][0])
            ax[pair_idx*2].imshow(cv2.cvtColor(images_and_crops[pair_idx][0], cv2.COLOR_BGR2RGB))
            ax[pair_idx*2+1].imshow(cv2.cvtColor(images_and_crops[pair_idx][1][mask_Idx], cv2.COLOR_BGR2RGB))
            ax[pair_idx*2].axis('off')
            ax[pair_idx*2+1].axis('off')
        name = ""
        for variationIdx in range(0, len(images_and_crops[pair_idx][3][mask_Idx]),1):
            for word in range(1,len(images_and_crops[pair_idx][3][mask_Idx][variationIdx]),2):
                name = name + str(images_and_crops[pair_idx][3][mask_Idx][variationIdx][word])

        fig.suptitle(images_and_crops[pair_idx][2][mask_Idx] + name,fontsize=16)
        plt.show()
"""