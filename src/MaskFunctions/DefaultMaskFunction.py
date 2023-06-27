import cv2
import numpy as np
from matplotlib import pyplot as plt
def CalculateStep(step,stepsize,bottom_Value):
    return step * stepsize + bottom_Value

def DefaultMaskFunction(image, variableSteps, bottom_and_top):



    # Holds the values for
    step_Values = []
    for valueIdx in range(len(bottom_and_top)):
        step_Values.append([])

    for valueIdx in range (0,len(bottom_and_top),1):
        if variableSteps[valueIdx] <= 1:
            if not isinstance(bottom_and_top[valueIdx],list):
                value = bottom_and_top[valueIdx]
                step_Values[valueIdx].append(value)
            else:
                raise Exception("Only one value [singleValue] was given yet steps are not 1")
        else:
            if  isinstance(bottom_and_top[valueIdx],list):
                value_bottom = bottom_and_top[valueIdx][0]
                value_top = bottom_and_top[valueIdx][1]
                stepsize = (value_top - value_bottom) /(variableSteps[valueIdx] - 1)

                step = 0
                while True:
                    step_Values[valueIdx].append(CalculateStep(step, stepsize, value_bottom))
                    step = step +1
                    if step >= variableSteps[valueIdx]:
                        break
            else:
                raise Exception("A top and a bottom value [or a list of 1 value] were given for the variable, yet there are is only 1 step")


    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finmask = np.where(mask < 255, 0, 1)

    used_variables = []
    mask_variations = []

    for value1 in step_Values[0]:
        for value2 in step_Values[1]:
            variable_to_append = []
            variable_to_append.append(["Variablename 1:"])
            if isinstance(value1, list):
                variable_to_append.append(value1[0])
            else:
                variable_to_append.append(value1)
            variable_to_append.append(["Variablename 2:"])
            if isinstance(value2, list):
                variable_to_append.append(value2[0])
            else:
                variable_to_append.append(value2)

            used_variables.append(variable_to_append)
            mask_variations.append(finmask)

    """
    for immask in mask_variations:
        plt.imshow(immask)
        plt.show()
        plt.imshow(image)
        plt.show()
    """
    return  used_variables, mask_variations
