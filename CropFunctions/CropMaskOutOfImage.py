import cv2
def CropMaskOutOfImage(image):
    print("hi")
    #if (image == rgb_image).all():
    #    print('The image was in the BGR colorspace')
    #else:
    #    print('The image was not in the BGR colorspace')
    croppedImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return croppedImage
    #TODO: And this