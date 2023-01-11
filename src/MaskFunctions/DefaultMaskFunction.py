import cv2
import numpy as np
from matplotlib import pyplot as plt
def CalculateStep(step,stepsize,bottom_Value):
    return step * stepsize + bottom_Value

def DefaultMaskFunction(image, variableSteps, bottom_and_top):

    print(bottom_and_top)
    if not isinstance(bottom_and_top[0]):
        bottom_Value1 = bottom_and_top[0]
        max_Value1 = bottom_and_top[0]
        if variableSteps[0]
    else:
        bottom_Value1 = bottom_and_top[0][0]
        max_Value1 = bottom_and_top[0][1]

    if not isinstance(bottom_and_top[1]):
        bottom_Value2 = bottom_and_top[1]
        max_Value2 = bottom_and_top[1]
    else:
        bottom_Value2 = bottom_and_top[1][0]
        max_Value2 = bottom_and_top[1][1]

    stepsize1 = (max_Value1-bottom_Value1)/max([(variableSteps[0]),1])

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finmask = np.where(mask < 255, 0, 1)

    stepsize2 = (max_Value2-bottom_Value2)/max([(variableSteps[1]),1])

    used_variables = []
    mask_variations = []

    step1 = 0
    while True:
        step2 = 0
        while True:
            calculated_step1 = CalculateStep(step1, stepsize1, bottom_Value1)
            calculated_step2 = CalculateStep(step2, stepsize2, bottom_Value2)
            #print(calculated_step1)
            #print(calculated_step2)
            used_variables.append([["Variablename 1:"], [calculated_step1], ["Variablename 2:"], calculated_step2])
            mask_variations.append(finmask)
            step2 = step2 + 1
            if step2 == variableSteps[1]:
                break
        step1 = step1 + 1
        if step1 == variableSteps[0]:
            break


    print(used_variables)
    print("")
    """
    for immask in mask_variations:
        plt.imshow(immask)
        plt.show()
        plt.imshow(image)
        plt.show()
    """
    return  used_variables, mask_variations