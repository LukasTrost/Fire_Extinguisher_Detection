import cv2
import os
import openpyxl

# checks whether a Folder for results exists, if not creates it
def CreateFolderForResults(images_and_values, datapath_cropped = ""):
    # Set a counter to track the number of attempts
    counter = 1
    datapath = datapath_cropped

    # Keep trying until file name doesnt exist

    while os.path.exists(datapath):
        datapath = datapath_cropped
        counter += 1
        # Append the attempt number to the path
        datapath = f"{datapath}{counter}"

    # Create the directory
    os.makedirs(datapath)
    print(f"The directory was created at " + datapath_cropped + str(counter))

    excelfile = openpyxl.Workbook()
    excelsheet = excelfile.active
    excelsheet.title = "Masks and Accuracies"
    excelsheet.cell(column=1, row=1, value = "Folder")

    # could be done in one nested loop but writing consecutive values is faster
    for idx in range(0, len(images_and_values), 1):
        excelsheet.cell(column=1, row=idx + 2, value=images_and_values[idx][6])


    for maskIdx in range(0, len(images_and_values[0][7]), 1):
        mask_folder_path = os.path.join(datapath,images_and_values[0][7][maskIdx])
        excelsheet.cell(column=maskIdx+2, row=1, value=images_and_values[0][7][maskIdx])
        os.makedirs(mask_folder_path)
        for idx in range(0, len(images_and_values), 1):
                new_folder_path = os.path.join(mask_folder_path, str(idx))
                os.makedirs(new_folder_path)
                cv2.imwrite(f"{new_folder_path}\\Original_image_"+images_and_values[idx][2]+ ".jpg", images_and_values[idx][0])
                cv2.imwrite(f"{new_folder_path}\\Cropped_"+images_and_values[idx][2]+".jpg", images_and_values[idx][1][maskIdx])
                cv2.imwrite(f"{new_folder_path}\\Mask_" + images_and_values[idx][2]+".jpg", images_and_values[idx][3][maskIdx])
                cv2.imwrite(f"{new_folder_path}\\Centered_" + images_and_values[idx][2]+".jpg", images_and_values[idx][4][maskIdx])
                excelsheet.cell(column=maskIdx+2, row=idx+2, value = images_and_values[idx][5][maskIdx])

    print(os.path.join(datapath,"Results.xlsx"))
    excelfile.save(f'{os.path.join(datapath,"Results.xlsx")}')
    print(f"The directory was filled.")