import os
import cv2
from src.ResizeFunctions.ResizeImage import ResizeFunction
from matplotlib import pyplot as plt
if __name__ == '__main__':
    file_path = "D:\Programmieren\MasterOfDisaster\Implementierungen\TestImages_and_Results\OriginalImages"
    images = os.listdir(file_path)

    for idx,image in enumerate(images):
        if idx <1:
            resized_image = ResizeFunction(cv2.imread(os.path.join(file_path, image)), (640,640))
            for colorspace in range(resized_image.shape[2]):
                R_G_or_B_image = resized_image[:,:,colorspace]
                plt.imshow(R_G_or_B_image)
                plt.show()
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.show()

    print(os.listdir(file_path))
    print(file_path)
