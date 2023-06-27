import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import math
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import skimage.measure
import os


def closest_pixel_contour_distance(shape, contour1, contour2):
    mask1 = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    mask2 = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)
    points1 = cv2.findNonZero(mask1)
    # print(points1)
    # print("placeholder")
    points1 = points1.reshape(-1, 2)
    # print(points1)

    points2 = cv2.findNonZero(mask2)
    points2 = points2.reshape(-1, 2)
    distances = cdist(points1, points2, 'euclidean')
    min_distance = np.min(distances)

    # print(min_distance)
    # plt.imshow(mask1)
    # plt.show()
    # plt.imshow(mask2)
    # plt.show()
    """
    min_distance = float('inf')
    for point1 in contour1:
        for point2 in contour2:
            #first value is x/y
            print(point1)
            #print(point2)
            #print(contour1)
            print(contour2)
            dist = cv2.pointPolygonTest(contour2, (int(point1[0][0]),int(point1[0][1])), True)
            min_distance = min(min_distance, dist)
            print(dist)
            dist = cv2.pointPolygonTest(np.array(contour1), (int(point2[0][0]),int(point2[0][1])), True)
            min_distance = min(min_distance, dist)
    """
    """
            distance = np.linalg.norm(point1[0]-point2[0])
            if distance < min_distance:
                min_distance = distance
    """
    return min_distance


def canny(img, blurParameters, cannyParameters):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blurParameters, 0)
    canny = cv2.Canny(blur, cannyParameters[0], cannyParameters[1])
    return canny


def regiongrowing(img, HSV_img, detected_contourgroup_or_groups, maximal_amount_of_rounds=30, blurParameters=[5, 5],
                  cannyParameters=[50, 200], cannyParametersStepSizes=[10, 10], kernelparameters=(3, 3),
                  colorthreshold=5):
    height, width, channels = img.shape
    kernel = np.ones(kernelparameters, np.uint8)
    if kernelparameters[0] == kernelparameters[1] and kernelparameters[0] % 2 == 1:
        kerneldirection = math.floor(kernelparameters[0] / 2)
        print("ther are " + str(len(detected_contourgroup_or_groups)) + "objects detected")
        reg_grown_contours = []
        for idx, contourgroup in enumerate(detected_contourgroup_or_groups):
            print("there are " + str(len(contourgroup)) + "contours in this object ")

            # temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            # cv2.drawContours(temp_img, contourgroup, -1, 255, -1)
            # plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
            # plt.show()
            DebugpixelAddedCounter = 0

            iterationCounter = 0
            contourmask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contourmask, contourgroup, -1, 255, -1)
            new_pixels_added = True

            edges = canny(img, blurParameters, [cannyParameters[0], cannyParameters[1]])
            ret, mask = cv2.threshold(edges, 2, 10000, cv2.THRESH_BINARY)
            mask = cv2.bitwise_not(mask)
            borderImage = cv2.bitwise_and(HSV_img, HSV_img, mask=mask)
            # plt.imshow(mask)
            # plt.show()

            parameteriterator = 1
            while new_pixels_added:
                # print("Iteration: "+str(iterationCounter) + " of: "+ str(maximal_amount_of_rounds))
                if iterationCounter >= maximal_amount_of_rounds:
                    # plt.imshow(contourmask)
                    # plt.show()

                    iterationCounter = 0
                    contourmask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(contourmask, contourgroup, -1, 255, -1)
                    if (cannyParameters[0] + parameteriterator * cannyParametersStepSizes[0]) > 255 and (
                            cannyParameters[1] + parameteriterator * cannyParametersStepSizes[1] > 255):
                        # canny parameters can be of whatever size since they are not restrained to 255 images
                        # so no problem if one overshoots while one is still being increased
                        print("region growing keeps failing, because borders are not pronounced enough")
                        reg_grown_contours.append(contourgroup)
                        break

                    edges = canny(img, blurParameters,
                                  [cannyParameters[0] + parameteriterator * cannyParametersStepSizes[0],
                                   cannyParameters[1] + parameteriterator * cannyParametersStepSizes[1]])
                    parameteriterator = parameteriterator + 1
                    ret, mask = cv2.threshold(edges, 2, 10000, cv2.THRESH_BINARY)
                    mask = cv2.bitwise_not(mask)
                    borderImage = cv2.bitwise_and(HSV_img, HSV_img, mask=mask)

                # Dilate the red_mask to get a larger region around the red pixels
                dilated_contour_mask = cv2.dilate(contourmask, kernel, iterations=1)
                # Find the borders of the red mask
                borders = cv2.bitwise_xor(dilated_contour_mask, contourmask)

                new_pixels_added = False
                add_to_mask = False
                for i in range(borders.shape[0]):
                    for j in range(borders.shape[1]):
                        # If the pixel is part of the borders
                        if borders[i, j] != 0:
                            # Get the color of the pixel
                            color = borderImage[i, j]
                            # Initialize a flag to indicate whether the pixel should be added to the red_mask
                            add_to_mask = False
                            # Iterate over the 3x3 neighborhood of the pixel but ignore the pixel in the middle
                            for x in range(i - kerneldirection, i + kerneldirection, 1):
                                for y in range(j - kerneldirection, j + kerneldirection, 1):
                                    # catch the case where the middle of the kernel is checked as well
                                    if not x == i or not y == j:

                                        # If the pixel is  part of the red_mask
                                        if contourmask[x, y] > 0:
                                            # Get the color of the mask pixel
                                            neighbor_color = borderImage[x, y]
                                            # Calculate the euclidean distance between the maskpixel and its neighbor
                                            diff = np.sqrt(np.sum((color - neighbor_color) ** 2))

                                            # if diff == 0:
                                            #    print(str(i) + " " + str(j) + " " + str(x) + " " + str(y) )

                                            # If the color difference is above the threshold
                                            if diff < colorthreshold:
                                                DebugpixelAddedCounter = DebugpixelAddedCounter + 1
                                                # Set the flag to add the pixel to the red_mask
                                                add_to_mask = True
                                                break
                                if add_to_mask:
                                    break
                            # If the flag is set, add the pixel to the red_mask
                            if add_to_mask:
                                contourmask[i, j] = 255
                                new_pixels_added = True
                                # print("I: " + str(i) + " and J: " + str(j) + "added")
                iterationCounter = iterationCounter + 1
            print("Overall " + str(DebugpixelAddedCounter) + " Pixels were added")
            contours, _ = cv2.findContours(contourmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            reg_grown_contours.append(contours)

            # temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            # cv2.drawContours(temp_img, contours, -1, 255, -1)
            # plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
            # plt.show()
        return reg_grown_contours
    else:
        print("kernel was either not symmetrical or uneven, Region growing was aborted")
        return detected_contourgroup_or_groups


def find_smaller_side(array_of_contour_lists):
    smaller_side_max_size = 0
    # loop through the array of lists of contours
    for i, contours in enumerate(array_of_contour_lists):
        # find the dimensions of each contourgroup
        points = np.vstack(contours)
        points = np.array(points, dtype=np.int32)
        rect = cv2.minAreaRect(points)
        width, height = rect[1]
        # print("x = "+ str(x) + "y = "+ str(y) + "w = "+ str(w) + "h ="+ str(h))

        # update the max_size and max_index if the current size is bigger
        if min(width, height) > smaller_side_max_size:
            smaller_side_max_size = min(width, height)

    return smaller_side_max_size


def fill_holes_in_mask(mask):
    # plt.imshow(mask)
    # plt.show()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)
        # plt.imshow(mask)
    # plt.show()
    return mask


def detect_objects_in_mask_and_merge_contours(mask, threshold_multiplicator1=0.0010175,
                                              threshold_multiplicator2=0.0010175):
    # Find the contours of the red object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []
    detected_objects.append([contours[0]])
    contours = contours[1:]
    # remove first contour and use as starting point

    threshold = 0

    for contour in contours:
        size = cv2.contourArea(contour)
        threshold = max(threshold, size)
        # print("first Threshold = "+ str(threshold))

    # Threshold is dependendant on size of biggest contour, so variable size of object is accounted for
    threshold = threshold * threshold_multiplicator1
    # print("first Threshold = "+ str(threshold))
    # check all contours of each detected object, if they are physically close enough to next contour in list
    # add that contour to that object, if no objects match create new object
    while len(contours) > 0:
        breakcondition = False
        for x in range(0, len(detected_objects), 1):
            for contour in detected_objects[x]:
                if closest_pixel_contour_distance(mask.shape, contours[0], contour) < threshold:
                    detected_objects[x].append(contours[0])
                    # print(closest_pixel_contour_distance(mask.shape,contours[0], contour))
                    contours = contours[1:]
                    breakcondition = True
                    break
                if breakcondition:
                    break
            if breakcondition:
                break
        if not breakcondition:
            detected_objects.append([contours[0]])
            contours = contours[1:]

    # re calculate threshold, since now it is most likely more accurate
    for contourobject in detected_objects:
        size = sum([cv2.contourArea(c) for c in contourobject])
        threshold = max(threshold, size)
        # print("second Threshold = "+ str(threshold))
    # print("second Threshold = "+ str(threshold))
    threshold = threshold * threshold_multiplicator2
    # print("second Threshold = "+ str(threshold))

    mergecondition = True
    newbreakcondition = False
    efficiencyCounter = 0
    iterator = 0
    while mergecondition:
        newbreakcondition = False
        for idx in range(iterator, len(detected_objects), 1):
            for contour in detected_objects[idx]:
                for other_group_idx in range(idx + 1, len(detected_objects), 1):
                    for othercontour in detected_objects[other_group_idx]:
                        if closest_pixel_contour_distance(mask.shape, contour, othercontour) < threshold:

                            """
                            #Debug
                            print("merged, new object, distance was =  " + str(closest_pixel_contour_distance(mask.shape,contour, othercontour)) )
                            temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                            cv2.drawContours(temp_img, detected_objects[idx], -1, 255, -1)
                            plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
                            plt.show() 
                            temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                            cv2.drawContours(temp_img, detected_objects[other_group_idx], -1, 255, -1)
                            plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
                            plt.show() 
                            """

                            for newothercontour in detected_objects[other_group_idx]:
                                detected_objects[idx].append(newothercontour)
                            newbreakcondition = True
                            if other_group_idx < len(detected_objects) - 1:
                                detected_objects = detected_objects[:other_group_idx] + detected_objects[
                                                                                        (other_group_idx + 1):]
                            else:
                                detected_objects = detected_objects[:len(detected_objects) - 1]
                        # else:
                        #    print(closest_pixel_contour_distance(mask.shape,contour, othercontour))
                        if newbreakcondition:
                            break
                    if newbreakcondition:
                        break
                if newbreakcondition:
                    break
            if newbreakcondition:
                break
            else:
                efficiencyCounter = efficiencyCounter + 1
        iterator = iterator + efficiencyCounter
        if not newbreakcondition:
            mergecondition = False
    return detected_objects


def count_neighbors(mask, area_where_to_count, neighborhood_size):
    # Create a zero array to store the count of 255 pixels for each pixel
    counts = np.zeros_like(mask, dtype=int)

    # Get height and width of the mask
    if neighborhood_size % 2 != 1:
        neighborhood_size = neighborhood_size + 1
        print("adjusted kernel since it was not divisible by two")

    neighborhood_length = math.floor(neighborhood_size / 2)
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            # Count the number of 255 pixels in the XxX neighborhood
            count = 0
            for x in range(-neighborhood_length, neighborhood_length, 1):
                for y in range(-neighborhood_length, neighborhood_length, 1):
                    if x + i > 0 and x + i < h and j + y > 0 and j + y < w:
                        if x != 0 or y != 0:
                            if area_where_to_count[x + i, j + y] == 255:
                                if mask[x + i, j + y] == 255:
                                    count += 1
            counts[i, j] = count
    return counts


def create_color_mask(hsv_image, array_of_masks):
    ranges = []

    for values in array_of_masks:
        lower = np.array(values[0])
        upper = np.array(values[1])
        ranges.append(cv2.inRange(hsv_image, lower, upper))

    mask = ranges[0]
    for rangeinstance in ranges:
        mask = mask | rangeinstance

    # plt.imshow(mask)
    # plt.show()
    return mask


def closest_color(pixel, color_list):
    # Find the closest color in the list to the given pixel.
    min_distance = float("inf")
    closest_color = None
    for color in color_list:
        distance = np.sum((pixel - color) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color


def create_extra_object_check_mask(img, objects_mask, neighbors_to_count=3, area_to_check_size=113,
                                   area_to_check_multiplier=0.3):
    kernel_size = int(area_to_check_size * area_to_check_multiplier)
    # print(kernel_size)
    if kernel_size % 2 != 1:
        kernel_size = kernel_size + 1

    dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(objects_mask, dilate_kernel, iterations=1)
    area_around_mask = cv2.subtract(dilated_mask, objects_mask)
    return area_around_mask


def check_for_busy_surroundings_around_object(img, objects_mask, area_around_mask,
                                              correction_dilation_kernel_parameters=[3, 3], blurParameters=[5, 5],
                                              cannyParameters=[80, 110]):
    # print("original image")
    # plt.imshow(img)
    # plt.show()
    # print("border_mask")
    # plt.imshow(area_around_mask)
    # plt.show()
    edges = canny(img, blurParameters, [cannyParameters[0], cannyParameters[1]])
    ret, mask = cv2.threshold(edges, 2, 10000, cv2.THRESH_BINARY)

    correction_dilation_kernel = np.ones(
        (correction_dilation_kernel_parameters[0], correction_dilation_kernel_parameters[1]), np.uint8)
    correction_mask = cv2.dilate(objects_mask, correction_dilation_kernel, iterations=1)

    # print("edge_picture")
    # plt.imshow(mask)
    # plt.show()
    canny_mask_minus_objects = copy.deepcopy(mask)
    canny_mask_minus_objects[correction_mask == 255] = 0
    # print("canny_mask_minus_objects")
    # print("This enough?")

    # print("area around object")
    # count_image = count_neighbors(mask = canny_mask_minus_objects, area_where_to_count = area_around_mask, neighborhood_size = neighbors_to_count)
    count_image = retrieve_structures(mask=canny_mask_minus_objects, area_where_to_count=area_around_mask)
    # plt.imshow(count_image)
    # plt.show()
    return count_image


def retrieve_structures(mask, area_where_to_count):
    return cv2.bitwise_and(mask, mask, mask=area_where_to_count)


def create_image_and_draw_contours_into_it(img_dimensions, contours, printIt=False):
    temp_img = np.zeros((img_dimensions[0], img_dimensions[1]), dtype=np.uint8)
    cv2.drawContours(temp_img, contours, -1, 255, -1)
    if printIt:
        plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
        plt.show()
    return temp_img


def change_image_colors(image, color_list):
    # Replace each pixel in the image with the closest color in the list
    new_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            closest = closest_color(image[i, j], color_list)
            # print(closest)
            new_image[i, j] = closest
    # print(new_image[i,j])
    return new_image


def determine_color_range_of_surroundings(img, hsv_image, area_around_mask, amount_clusters):
    dummy_img = img
    dummy_img = dummy_img.reshape((dummy_img.shape[1] * dummy_img.shape[0], 3))
    mask = area_around_mask
    mask = mask.reshape((mask.shape[1] * mask.shape[0]))
    idxes = []
    for pixelidx in range(len(dummy_img)):
        if mask[pixelidx] == 0:
            idxes.append(pixelidx)
    dummy_img = np.delete(dummy_img, idxes, 0)

    kmeans = KMeans(n_clusters=amount_clusters, init='k-means++', random_state=0)

    s = kmeans.fit(dummy_img)

    labels = kmeans.labels_
    labels = list(labels)
    centroids = kmeans.cluster_centers_
    # print(centroids)

    percent = []
    for i in range(len(centroids)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)

    # plt.pie(percent,colors=np.array(centroids/255),labels=np.arange(len(centroids)))
    # plt.show()

    filtered_img = change_image_colors(img, centroids)
    filtered_img = cv2.bitwise_and(filtered_img, filtered_img, mask=area_around_mask)
    # plt.imshow(filtered_img)
    # plt.show()
    # [198.35363025 206.40782301 210.31141267]
    color_masks = []
    min_size_percent = 0.1
    for centroid in centroids:
        # print(centroid)
        lower = np.array([int(centroid[0]), int(centroid[1]), int(centroid[2])])
        # print(lower)
        upper = lower
        mask = cv2.inRange(filtered_img, lower, upper)

        # 8 is the neighborhood
        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

        # Split the binary mask into individual masks, one for each object
        masks = []
        # print("0")
        # print(output[0])
        # print("1")
        # print(output[1])
        # print("2")
        # print(output[2])
        # print("3")
        # print(output[3])
        labels = output[1]
        stats = output[2]
        max_area = 0
        # first entry for some reason stores some other stuff so ignore it for loops
        for i in range(1, output[0]):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area

        for i in range(1, output[0]):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= max_area * min_size_percent:
                new_mask = np.zeros_like(labels)
                new_mask[labels == i] = 255
                masks.append(new_mask)
                # plt.imshow(new_mask)
                # plt.show()

        # Store the masks for this color
        color_masks.append(masks)

        # plt.imshow(mask)
        # plt.show()
        # display_img = copy.deepcopy(filtered_img)
        # plt.imshow(cv2.bitwise_and(display_img, display_img, mask=mask))
        # plt.show()

    return color_masks


def combine_structure_and_colors(color_masks, surround_mask, acceptance_threshold):
    # print(color_masks[0])
    # print(color_masks[0][0])
    objects_to_analyze = []
    for color_objects in color_masks:
        for colored_object in color_objects:
            # print(colored_object.shape)
            # print(colored_object)
            # plt.imshow(colored_object)
            # plt.show()
            colored_obj = np.uint8(colored_object)
            _, thresh_mask = cv2.threshold(colored_obj, 1, 255, cv2.THRESH_BINARY)

            dilation_kernel = np.ones((3, 3), np.uint8)
            correction_mask = cv2.dilate(thresh_mask, dilation_kernel, iterations=1)

            comp_mask = cv2.bitwise_and(surround_mask, correction_mask)
            non_zero_pixels = cv2.countNonZero(comp_mask)
            # use dilation to count structures, but original to count percentage
            total_pixels = cv2.countNonZero(thresh_mask)
            percent_covered = (non_zero_pixels / total_pixels) * 100

            # print("Percent covered:", percent_covered)
            if percent_covered > acceptance_threshold:
                # plt.imshow(colored_object)
                # plt.show()
                # plt.imshow(surround_mask)
                # plt.show()
                objects_to_analyze.append(colored_object)
    return objects_to_analyze


def identify_other_parts(img, hsv_image, objects_mask, correction_dilation_kernel_parameters=[3, 3],
                         blurParameters=[5, 5], cannyParameters=[80, 110], neighbors_to_count=3, area_to_check_size=113,
                         area_to_check_multiplier=0.3, acceptance_threshold=10.0, amount_clusters=8):
    mask_of_area_around_mask = create_extra_object_check_mask(img, objects_mask, neighbors_to_count, area_to_check_size,
                                                              area_to_check_multiplier)
    surround_mask = check_for_busy_surroundings_around_object(img, objects_mask,
                                                              correction_dilation_kernel_parameters=correction_dilation_kernel_parameters,
                                                              area_around_mask=mask_of_area_around_mask,
                                                              blurParameters=blurParameters,
                                                              cannyParameters=cannyParameters)
    color_objects = determine_color_range_of_surroundings(img=img, hsv_image=hsv_image,
                                                          area_around_mask=mask_of_area_around_mask,
                                                          amount_clusters=amount_clusters)
    objects_to_analyze = combine_structure_and_colors(color_objects, surround_mask, acceptance_threshold)
    return objects_to_analyze


def mask_regiongrowing(img, HSV_img, detected_object_mask_or_masks, maximal_amount_of_rounds_multiplier=1,
                       blurParameters=[5, 5], cannyParameters=[50, 200], cannyParametersStepSizes=[10, 10],
                       kernelparameters=(3, 3), colorthreshold=5):
    print("mask region growing")
    interrupter_counter = 0
    kernel = np.ones(kernelparameters, np.uint8)
    if kernelparameters[0] == kernelparameters[1] and kernelparameters[0] % 2 == 1:
        kerneldirection = math.floor(kernelparameters[0] / 2)
        print("ther are " + str(len(detected_object_mask_or_masks)) + "masks that will be region grown")
        reg_grown_masks = []
        for idx, object_mask in enumerate(detected_object_mask_or_masks):
            # print("another mask is growing")
            # print(object_mask)
            # plt.imshow(object_mask)
            # plt.show()
            iterationCounter = 0
            new_pixels_added = True
            current_cannyParameters = cannyParameters
            edges = canny(img, blurParameters, [current_cannyParameters[0], current_cannyParameters[1]])
            ret, mask = cv2.threshold(edges, 2, 10000, cv2.THRESH_BINARY)
            mask = cv2.bitwise_not(mask)
            borderImage = cv2.bitwise_and(HSV_img, HSV_img, mask=mask)
            # plt.imshow(mask)
            # plt.show()
            current_mask = np.uint8(object_mask)
            parameteriterator = 1

            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            smaller_side = find_smaller_side(contours)
            maximal_amount_of_rounds = smaller_side * maximal_amount_of_rounds_multiplier
            print("maximal amount of round = " + str(maximal_amount_of_rounds))

            while new_pixels_added:
                # print("Iteration: "+str(iterationCounter) + " of: "+ str(maximal_amount_of_rounds))
                if iterationCounter >= maximal_amount_of_rounds:
                    interrupter_counter = interrupter_counter  +1
                    print("interruptercounter mask region growing" , interrupter_counter)
                    # plt.imshow(contourmask)
                    # plt.show()

                    iterationCounter = 0
                    current_cannyParameters[0] = current_cannyParameters[0] - cannyParametersStepSizes[0]
                    if current_cannyParameters[0] < 0:
                        current_cannyParameters[0] = 0
                    current_cannyParameters[1] = current_cannyParameters[1] - cannyParametersStepSizes[1]
                    if current_cannyParameters[1] < 0:
                        current_cannyParameters[1] = 0

                    print(current_cannyParameters)
                    if (current_cannyParameters[0]) <= 0 and (current_cannyParameters[1]) <= 0:
                        # canny parameters can be of whatever size since they are not restrained to 255 images
                        # so no problem if one overshoots while one is still being increased
                        print("region growing keeps failing, because borders are not pronounced enough")
                        reg_grown_masks.append(current_mask)
                        break
                    current_mask = np.uint8(object_mask)
                    edges = canny(img, blurParameters, [current_cannyParameters[0], current_cannyParameters[1]])
                    ret, mask = cv2.threshold(edges, 2, 10000, cv2.THRESH_BINARY)
                    mask = cv2.bitwise_not(mask)
                    borderImage = cv2.bitwise_and(HSV_img, HSV_img, mask=mask)

                # Dilate the red_mask to get a larger region around the red pixels
                # print(type(current_mask))
                # print(current_mask.dtype)
                dilated_current_mask = cv2.dilate(current_mask, kernel, iterations=1)
                # Find the borders of the red mask
                borders = cv2.bitwise_xor(dilated_current_mask, current_mask)

                new_pixels_added = False
                add_to_mask = False
                for i in range(borders.shape[0]):
                    for j in range(borders.shape[1]):
                        # If the pixel is part of the borders
                        if borders[i, j] != 0:
                            # Get the color of the pixel
                            color = borderImage[i, j]
                            # Initialize a flag to indicate whether the pixel should be added to the mask
                            add_to_mask = False
                            # Iterate over the 3x3 neighborhood of the pixel but ignore the pixel in the middle
                            for x in range(i - kerneldirection, i + kerneldirection, 1):
                                for y in range(j - kerneldirection, j + kerneldirection, 1):
                                    # catch the case where the middle of the kernel is checked as well
                                    if not x == i or not y == j:

                                        # If the pixel is  part of the _mask
                                        if current_mask[x, y] > 0:
                                            # Get the color of the mask pixel
                                            neighbor_color = borderImage[x, y]
                                            # Calculate the euclidean distance between the maskpixel and its neighbor
                                            diff = np.sqrt(np.sum((color - neighbor_color) ** 2))

                                            # if diff == 0:
                                            #    print(str(i) + " " + str(j) + " " + str(x) + " " + str(y) )

                                            # If the color difference is above the threshold
                                            if diff < colorthreshold:
                                                # Set the flag to add the pixel to the red_mask
                                                add_to_mask = True
                                                break
                                if add_to_mask:
                                    break
                            # If the flag is set, add the pixel to the red_mask
                            if add_to_mask:
                                current_mask[i, j] = 255
                                new_pixels_added = True
                                # print("I: " + str(i) + " and J: " + str(j) + "added")
                iterationCounter = iterationCounter + 1
            reg_grown_masks.append(current_mask)

        return reg_grown_masks
    else:
        print("kernel was either not symmetrical or uneven, Region growing was aborted")
        return detected_object_mask_or_masks


def efficient_pixel_distance(small_mask, big_mask, kerneliterations):
    kernel = np.ones([3, 3], np.uint8)
    dilated_mask = cv2.dilate(small_mask, kernel, iterations=kerneliterations)
    if (dilated_mask & big_mask).any():
        return True
    return False


def approximate_hsv_range(hsv_img, mask=None):
    if mask is not None and mask.any():
        mean, std = cv2.meanStdDev(hsv_img, mask=mask)
    else:
        mean, std = cv2.meanStdDev(hsv_img)
    # maximum for HSV is 180, 255, 255
    # Hue is cyclical, Saturation and Value are not
    # so clamping has to be done differently
    # print("mean" + str(mean))
    # print("std" + str(std))
    H_Lower = mean[0][0] - 2 * std[0][0]
    H_Upper = mean[0][0] + 2 * std[0][0]

    S_Lower = mean[1][0] - 2 * std[1][0]
    # print(S_Lower)
    if S_Lower < 0:
        S_Lower = 0
    # print(S_Lower)
    S_Upper = mean[1][0] + 2 * std[1][0]
    if S_Upper > 255:
        S_Upper = 255

    V_Lower = mean[2][0] - 2 * std[2][0]
    if V_Lower < 0:
        V_Lower = 0
    V_Upper = mean[2][0] + 2 * std[2][0]
    if V_Upper > 255:
        V_Upper = 255
    lower = np.array([H_Lower, S_Lower, V_Lower])
    upper = np.array([H_Upper, S_Upper, V_Upper])
    return lower, upper


def binary_region_growing(mask_to_grow, borders_and_shapes_mask, rounds):
    plt.imshow(mask_to_grow)
    plt.show()
    plt.imshow(borders_and_shapes_mask)
    plt.show()
    print("They will grow for this amount of rounds: ", rounds)
    rounds = 50
    grown_mask = copy.deepcopy(mask_to_grow)
    kernel = np.ones([3, 3], np.uint8)
    # Cant use x9 neighborhood here, since then borders might be crossed accidently
    kernel[0, 0] = 0
    kernel[0, 2] = 0
    kernel[2, 0] = 0
    kernel[2, 2] = 0
    cv2.imwrite("borders_and_shapes_mask.jpg", borders_and_shapes_mask)
    for current_round in range(rounds):
        print(cv2.countNonZero(grown_mask))

        dilated_mask = cv2.dilate(grown_mask, kernel, iterations=1)
        borders = cv2.bitwise_xor(dilated_mask, grown_mask)
        plt.imshow(borders)
        plt.show()
        plt.imshow(borders_and_shapes_mask)
        plt.show()
        new_pixels_added = False

        for x in range(borders.shape[0]):
            for y in range(borders.shape[1]):
                if borders[x, y] == 255 and grown_mask[x, y] == 0 and borders_and_shapes_mask[x, y] == 0:
                    grown_mask[x, y] = 255
                    new_pixels_added = True

        if not new_pixels_added:
            # readd the borders to the mask
            grown_mask = cv2.dilate(grown_mask, kernel, iterations=1)
            plt.imshow(grown_mask)
            plt.show()
            print("it took ", current_round, " rounds to grow")
            return grown_mask, True
        plt.imshow(grown_mask)
        plt.show()
    print("growing aborted")
    cv2.imwrite("grown_mask.jpg", grown_mask)
    return grown_mask, False


def check_what_parts_of_mask_are_surrounded(img, combined_mask, structure_to_test, blurParameters=[5, 5],
                                            cannyParameters=[80, 110], cannyParametersStepsize=[30, 30],
                                            size_multiplier=0.5, max_NonZero_multiplier=0.01):
    edges = canny(img, blurParameters, [cannyParameters[0], cannyParameters[1]])
    ret, mask = cv2.threshold(edges, 2, 255, cv2.THRESH_BINARY)
    # mask = cv2.bitwise_not(mask)

    # print("canny edges")
    # plt.imshow(mask)
    # plt.show()

    # print("extinguisher_mask")
    # plt.imshow(combined_mask)
    # plt.show()

    mask_and_edges = cv2.bitwise_or(mask, combined_mask)
    # print("both")
    # plt.imshow(mask_and_edges)
    # plt.show()

    mask_and_edges = cv2.bitwise_not(mask_and_edges)
    # print("both inverted")
    # plt.imshow(mask_and_edges)
    # plt.show()

    # print("following structure will be tested")
    temp_img = np.zeros_like(mask_and_edges)
    cv2.drawContours(temp_img, [structure_to_test], -1, (255, 255, 255), -1)
    # plt.imshow(temp_img)
    # plt.show()

    cut_mask = cv2.bitwise_or(temp_img, temp_img, mask=mask_and_edges)
    labeled_image = skimage.measure.label(cut_mask, connectivity=1)

    # Count number of connected components
    num_components = labeled_image.max()

    # Extract masks for each connected component
    component_masks = []
    for i in range(1, num_components + 1):
        component_mask = labeled_image == i
        component_mask = component_mask.astype('uint8') * 255
        component_masks.append(component_mask)
        # plt.imshow(component_mask)
        # plt.show()

    max_value = 0
    for mask_to_grow in component_masks:
        size = cv2.countNonZero(mask_to_grow)
        if size > max_value:
            max_value = size

    masks_to_add = []
    for mask_to_grow in component_masks:
        size = cv2.countNonZero(mask_to_grow)
        rounds = math.ceil(size_multiplier * size)
        rounds = max(rounds, max_value * max_NonZero_multiplier)
        print("this would grow for: ", rounds, " rounds")
        parameters = copy.deepcopy(cannyParameters)
        breakCondition = False
        new_edges = canny(img, blurParameters, [parameters[0], parameters[1]])
        new_ret, new_mask = cv2.threshold(new_edges, 1, 255, cv2.THRESH_BINARY)
        new_mask = cv2.bitwise_or(new_mask, combined_mask)
        dilate_counter = 1
        kernel = np.ones([3, 3], np.uint8)
        while True:
            grown_partial_mask, breakCondition = binary_region_growing(mask_to_grow, new_mask, rounds)
            if breakCondition:
                masks_to_add.append(grown_partial_mask)
                break
            new_edges = canny(img, blurParameters, [parameters[0], parameters[1]])
            new_ret, new_mask = cv2.threshold(new_edges, 1, 255, cv2.THRESH_BINARY)
            new_mask = cv2.dilate(new_mask, kernel, iterations=dilate_counter)
            new_mask = cv2.bitwise_or(new_mask, combined_mask)

    final_mask = np.zeros_like(mask_and_edges)
    for mask_to_finalize in masks_to_add:
        final_mask = cv2.bitwise_or(final_mask, mask_to_finalize)
        # cv2.findNonZero(mask1)
    return final_mask
    # plt.imshow(final_mask)
    # plt.show()

    """
    kernel_size = int(area_to_check_size * area_to_check_multiplier)
    #print(kernel_size)
    if kernel_size%2 != 1:
        kernel_size = kernel_size + 1

    dilate_kernel = np.ones((kernel_size,kernel_size),np.uint8)
    dilated_mask = cv2.dilate(objects_mask,dilate_kernel,iterations = 1)
    area_around_mask = cv2.subtract(dilated_mask, objects_mask)
    return area_around_mask
    """


def find_suitable_contours_around_color_contour(img, original_mask, mask, combined_mask, size_multiplier_contours,
                                                distance_multiplier, blurParameters=[5, 5], cannyParameters=[80, 110],
                                                cannyParametersStepsize=[30, 30], size_multiplier_region=0.5,
                                                max_NonZero_multiplier=0.01):  # threshold_multiplicator1 = 0.0010175, threshold_multiplicator2 = 0.0010175):

    original_contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(original_contours)
    original_area = sum([cv2.contourArea(c) for c in original_contours])
    size_threshold = original_area * size_multiplier_contours
    distance_threshold = original_area * distance_multiplier
    if distance_threshold < 1:
        distance_threshold = 1
    print("original area is")
    print(original_area)
    print("resulting distance threshold is")
    print(distance_threshold)
    # Find the contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if size_threshold < area:
            kept_contours.append(contour)
    # print("original mask")
    # temp_img = np.zeros((original_mask.shape[0], original_mask.shape[1]), dtype=np.uint8)
    # cv2.drawContours(temp_img, original_contours, -1, (255, 255, 255), -1)
    # plt.imshow(temp_img)
    # plt.show()
    for contour in kept_contours:
        print("contours that might be added")
        temp_img = np.zeros((original_mask.shape[0], original_mask.shape[1]), dtype=np.uint8)
        cv2.drawContours(temp_img, [contour], -1, (255, 255, 255), -1)
        plt.imshow(temp_img)
        plt.show()

    none_added = False
    break_condition = False
    while not none_added:
        none_added = True
        break_condition = False
        for contour in kept_contours:
            for original_contour in original_contours:
                if closest_pixel_contour_distance(original_mask.shape, contour, original_contour) < distance_threshold:
                    check_what_parts_of_mask_are_surrounded(img, combined_mask, contour, blurParameters=blurParameters,
                                                            cannyParameters=cannyParameters,
                                                            cannyParametersStepsize=cannyParametersStepsize,
                                                            size_multiplier=size_multiplier_region,
                                                            max_NonZero_multiplier=max_NonZero_multiplier)

                    merge_image = np.zeros_like(original_mask)
                    cv2.drawContours(merge_image, original_contours, -1, (255, 255, 255), -1)
                    cv2.drawContours(merge_image, [contour], -1, (255, 255, 255), -1)
                    original_contours, _ = cv2.findContours(merge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # print("intermediate result")
                    # plt.imshow(merge_image)
                    # plt.show()
                    kept_contours.remove(contour)
                    break_condition = True
                    none_added = False
                    break
                if break_condition:
                    break
            if break_condition:
                break
    print("final result")
    temp_img = np.zeros((original_mask.shape[0], original_mask.shape[1]), dtype=np.uint8)
    cv2.drawContours(temp_img, original_contours, -1, (255, 255, 255), -1)
    return temp_img
    # plt.imshow(temp_img)
    # plt.show()


def get_other_contours_closeby(hsv, img, object_mask, combined_mask, size_multiplier_contours, distance_multiplier,
                               blurParameters, cannyParameters, cannyParametersStepsize=[30, 30],
                               size_multiplier_region=0.5, max_NonZero_multiplier=0.01):
    # plt.imshow(object_mask)
    # plt.show()
    lower, upper = approximate_hsv_range(hsv, object_mask)
    # print("Lower is " +  str(lower))
    # print("Upper is " +  str(upper))
    mask = create_color_mask(hsv, [[lower, upper]])
    # plt.imshow(cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_BGR2RGB))
    # plt.show()
    return_mask = find_suitable_contours_around_color_contour(img, object_mask, mask, combined_mask,
                                                              size_multiplier_contours, distance_multiplier,
                                                              blurParameters, cannyParameters,
                                                              cannyParametersStepsize=cannyParametersStepsize,
                                                              size_multiplier_region=size_multiplier_region,
                                                              max_NonZero_multiplier=max_NonZero_multiplier)
    return return_mask
    # detect_objects_in_mask_and_merge_contours
    # check whether contours are in range to the object


def add_objects(img, hsv, big_mask, left_black_contours, correction_dilation_kernel_parameters=[11, 11],
                blurParameters_borderCheck=[5, 5], cannyParameters_borderCheck=[80, 110],
                blurParameters_identify=[5, 5],
                cannyParameters_identify=[80, 110], cannyParametersStepSizes_region=[10, 10],
                kernelparameters_region=(3, 3),
                blurParameters_region=[5, 5], colorthreshold_region=5,
                cannyParameters_region=[80, 110], neighbors_to_count=3, area_to_check_size=113,
                area_to_check_multiplier=0.3,
                amount_clusters=8, kerneliterations=1, maximal_amount_of_rounds_multiplier=0.2,
                size_multiplier_borderCheck_contours=0.2,
                distance_multiplier_borderCheck=0.015, size_multiplier_region_borderCheck=0.1,
                cannyParametersStepSizes_borderCheck=[30, 30],
                max_NonZero_multiplier_borderCheck=0.01):
    expanded_mask = copy.deepcopy(big_mask)
    something_added = True
    while something_added:
        something_added = False
        # print("First check for objects with lots of edges and append any that are in range")
        edgy_objects_to_append = identify_other_parts(img, hsv_image=hsv, objects_mask=expanded_mask,
                                                      correction_dilation_kernel_parameters=correction_dilation_kernel_parameters,
                                                      blurParameters=blurParameters_identify,
                                                      cannyParameters=cannyParameters_identify,
                                                      neighbors_to_count=neighbors_to_count,
                                                      area_to_check_size=area_to_check_size,
                                                      area_to_check_multiplier=area_to_check_multiplier,
                                                      amount_clusters=amount_clusters)

        for edgy_object_mask in edgy_objects_to_append:
            plt.imshow(edgy_object_mask)
            plt.show()
        # print(len(edgy_objects_to_append))
        print("Now mask region Growing")
        edgy_objects_to_append = mask_regiongrowing(img, hsv, edgy_objects_to_append,
                                                    maximal_amount_of_rounds_multiplier=maximal_amount_of_rounds_multiplier,
                                                    blurParameters=blurParameters_region,
                                                    cannyParameters=cannyParameters_region,
                                                    cannyParametersStepSizes=cannyParametersStepSizes_region,
                                                    kernelparameters=kernelparameters_region,
                                                    colorthreshold=colorthreshold_region)
        print("done mask region growing")
        # print("region growing done")
        for idx, object_mask in enumerate(edgy_objects_to_append):
            if (efficient_pixel_distance(object_mask, expanded_mask, kerneliterations)):
                # object_mask = get_other_contours_closeby(hsv, img, object_mask, expanded_mask,
                # size_multiplier_contours = size_multiplier_borderCheck_contours, distance_multiplier = distance_multiplier_borderCheck,
                # blurParameters = blurParameters_borderCheck, cannyParameters = cannyParameters_borderCheck,
                # cannyParametersStepsize = cannyParametersStepSizes_borderCheck, size_multiplier_region = size_multiplier_region_borderCheck,
                # max_NonZero_multiplier = max_NonZero_multiplier_borderCheck)

                expanded_mask = cv2.bitwise_or(object_mask, expanded_mask)
                something_added = True
                print("Added Something edgy")
        # print("One iteration Done")
        plt.imshow(expanded_mask)
        plt.show()

        """
        chosen_objects = []
        #check masks

        for idx, object_mask in enumerate(edgy_objects_to_append):
            if(efficient_pixel_distance(object_mask, expanded_mask, kerneliterations)):
                chosen_objects.append(object_mask)
                something_added = True
        # add the masks that were in range
        for chosen_mask in chosen_objects:
            expanded_mask = cv2.bitwise_or(chosen_mask[0], expanded_mask)
            edgy_objects_to_append.remove(chosen_mask)
        """
        # copy_list = left_black_contours[:]

        while True:
            something_black_added = False
            for black_contour in left_black_contours:
                # print("YO CHECK ME I M IN RANGE")
                temp_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.drawContours(temp_img, black_contour, -1, (255, 255, 255), -1)
                # plt.imshow(temp_img)
                # plt.show()

                if (efficient_pixel_distance(expanded_mask, temp_img, kerneliterations)):
                    expanded_mask = cv2.bitwise_or(temp_img, expanded_mask)
                    # print("all")
                    # print(type(left_black_contours))
                    # print("single")
                    # print(type(black_contour))
                    # print(len(left_black_contours))
                    left_black_contours = [c for c in left_black_contours if id(c) != id(black_contour)]
                    something_black_added = True
                    something_added = True
                    print("Added Something Black")
                    break
            if something_black_added == False:
                # print("Done adding black")
                break

        plt.imshow(expanded_mask)
        plt.show()
        # return expanded_mask

        # for metallic_mask in metallic_masks:

        # solange durch alle durch rotieren bis in einem durchlaufen nicht geadded wird
        # wenn nicht mehr kernelparameter übergeben sondern wie viele iterationen das dilatieren haben soll
        # anstelle von nur nemTrue / false direkt dann schon zu big mask adden, aber aufpassen das die originale nicht die dilattierte
        # ABER auch ein true, damit die maske dann schon aus der liste gekicked werden kann
        # oder doch erst distanz zueinander und dann zu object?
        # ne sollte so wie es ist schneller sein, da es wahrscheinlicher ist das etwas am großen objekt ist
        # distanzen alle auf pixel nicht kontouren?
        # distanz zwischen main rot und anderen rot einführen
        """

        black, after part check
        danach wieder kmeans, aber mit derselben maske
        solange wiederholen bis keine änderungen mehr
        in loop auch suche nach dem dritten einschliessen
        danach
        hoffentlich
        done
        """
    return expanded_mask  # new big mask
def CalculateStep(step,stepsize,bottom_Value):
    return step * stepsize + bottom_Value




def Perform_Extinguisher_Net(image,variables):

    img = image
    height, width, channels = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of colors for the red object
    # Red is at 0
    lower_red1 = np.array([170, 100, 100])
    upper_red1 = np.array([180, 255, 255])

    lower_red2 = np.array([0, 100, 100])
    upper_red2 = np.array([10, 255, 255])

    red_mask = create_color_mask(hsv, [[lower_red1, upper_red1], [lower_red2, upper_red2]])

    print("Step 1 Starting")
    detected_objects = detect_objects_in_mask_and_merge_contours(mask=red_mask, threshold_multiplicator1=0.0010175,
                                                                 threshold_multiplicator2=0.0009)

    smaller_side_max_size = find_smaller_side(detected_objects)

    # print("Maximum size of smaller side detected" + str(smaller_side_max_size))

    print("Step 2 Starting")
    reg_grow_detected_objects = regiongrowing(img, hsv, detected_objects,
                                              maximal_amount_of_rounds=int(smaller_side_max_size * 3 / 5),
                                              blurParameters=[5, 5], cannyParameters=[50, 200],
                                              cannyParametersStepSizes=[10, 10], kernelparameters=(3, 3),
                                              colorthreshold=5)

    smaller_side_max_size = find_smaller_side(reg_grow_detected_objects)

    # print("Maximum size of smaller side detected" + str(smaller_side_max_size))

    max_size = 0
    max_index = 0
    print("Step 3 Starting")
    # loop through the array of lists of contours
    for i, contours in enumerate(reg_grow_detected_objects):
        # find the size of each list of contours

        # contours are necessarily disjunct so calculating are of each individually is fine
        size = sum([cv2.contourArea(c) for c in contours])

        # update the max_size and max_index if the current size is bigger
        if size > max_size:
            max_size = size
            max_index = i

    # Do not append parts of red that are too far away from the main part
    # for i, contours in enumerate()
    print("Step 4 Starting")
    combined_red_contours_mask = np.zeros((height, width), dtype=np.uint8)

    # loop through the array of lists of contours
    for i, contours in enumerate(reg_grow_detected_objects):
        # find the size of each list of contours
        if i != max_index:
            cv2.drawContours(combined_red_contours_mask, contours, -1, 255, -1)
        else:
            points = np.vstack([c.squeeze() for c in reg_grow_detected_objects[i]])
            # Find the convex hull of all the points
            hull = cv2.convexHull(points)
            cv2.drawContours(combined_red_contours_mask, [hull], -1, 255, -1)
        # plt.imshow(combined_red_contours_mask)
        # plt.show()

    # plt.imshow(combined_red_contours_mask)
    # plt.show()

    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([180, 255, 60], dtype="uint8")

    black_mask = create_color_mask(hsv, [[lower_black, upper_black]])
    black_contours = detect_objects_in_mask_and_merge_contours(mask=black_mask, threshold_multiplicator1=0.0010175,
                                                               threshold_multiplicator2=0.00112)

    print("Step 5 starting")
    print(len(black_contours))
    contours = regiongrowing(img, hsv, black_contours, maximal_amount_of_rounds=int(smaller_side_max_size * 3 / 5),
                             blurParameters=[5, 5], cannyParameters=[50, 200], cannyParametersStepSizes=[50, 50],
                             kernelparameters=(3, 3), colorthreshold=5)

    new_contours = []
    discarded_contours = []

    # plt.imshow(combined_red_contours_mask)
    # plt.show()
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated_combined_red_contours_mask = cv2.dilate(combined_red_contours_mask, dilate_kernel, iterations=1)

    # plt.imshow(dilated_combined_red_contours_mask)
    # plt.show()
    # print(len(black_contours))
    print("Step 6 starting")
    for black_object in black_contours:
        temp_img = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(temp_img, black_object, -1, (255, 255, 255), -1)

        if (dilated_combined_red_contours_mask & temp_img).any():
            new_contours.append(black_object)
        else:
            discarded_contours.append(black_object)
    """
    #debug
    print("appended")
    for black_object in new_contours:
        temp_img = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(temp_img, black_object, -1, (255, 255, 255), -1)
        plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
        plt.show()


    print("discarded")
    for black_object in discarded_contours:
        temp_img = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(temp_img, black_object, -1, (255, 255, 255), -1)
        plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
        plt.show()
    #debug end
    """
    for new_contour in new_contours:
        cv2.drawContours(combined_red_contours_mask, new_contour, -1, (255, 255, 255), -1)

    print("Step 7 Starting")
    combined_red_contours_mask = fill_holes_in_mask(combined_red_contours_mask)

    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()

    #plt.imshow(cv2.cvtColor(combined_red_contours_mask, cv2.COLOR_BGR2RGB))
    #plt.show()

    result = cv2.bitwise_and(img, img, mask=combined_red_contours_mask)

    final_or_not_mask = add_objects(img, hsv=hsv, big_mask=combined_red_contours_mask,
                                    correction_dilation_kernel_parameters=[11, 11],
                                    blurParameters_borderCheck=[5, 5], cannyParameters_borderCheck=[80, 110],
                                    blurParameters_identify=[5, 5],
                                    cannyParameters_identify=[80, 110], cannyParametersStepSizes_region=[70, 70],
                                    kernelparameters_region=(3, 3),
                                    blurParameters_region=[5, 5], cannyParameters_region=[80, 110],
                                    colorthreshold_region=5, neighbors_to_count=3,
                                    area_to_check_size=smaller_side_max_size,
                                    area_to_check_multiplier=0.3, amount_clusters=8,
                                    left_black_contours=discarded_contours, kerneliterations=3,
                                    maximal_amount_of_rounds_multiplier=0.2, size_multiplier_borderCheck_contours=0.3,
                                    distance_multiplier_borderCheck=0.015, size_multiplier_region_borderCheck=0.5,
                                    cannyParametersStepSizes_borderCheck=[50, 50],
                                    max_NonZero_multiplier_borderCheck=0.01)
    final_or_not_mask = fill_holes_in_mask(final_or_not_mask)

    return final_or_not_mask


def Extinguisher_Net(image, variableSteps, bottom_and_top,address):
    # Holds the values for

    #print(bottom_and_top)
    #plt.imshow(image)
    #plt.show()
    #print(variableSteps)

    step_Values = []
    for valueIdx in range(len(bottom_and_top)):
        step_Values.append([])

    for valueIdx in range (0,len(bottom_and_top),1):
        #print()
        if variableSteps[valueIdx] <= 1:
            if not isinstance(bottom_and_top[valueIdx],list):
                value = bottom_and_top[valueIdx]
                step_Values[valueIdx].append(value)
            else:
                raise Exception("Only one value [singleValue] was given yet steps are not 1")
        else:
            if isinstance(bottom_and_top[valueIdx],list):
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

    print(step_Values)

    used_variables = []
    mask_variations = []



    for value1 in step_Values[0]:
        for value2 in step_Values[1]:
            for value3 in step_Values[2]:
                for value4 in step_Values[3]:
                    for value5 in step_Values[4]:
                        for value6 in step_Values[5]:
                            for value7 in step_Values[6]:
                                for value8 in step_Values[7]:
                                    print("performing Extinguisher_Net")

                                    variable_to_append = []
                                    performance_values = []

                                    variable_to_append.append(["lower_red_1_1:"])
                                    if isinstance(value1, list):
                                        variable_to_append.append(value1[0])
                                        performance_values.append(value1[0])

                                    else:
                                        variable_to_append.append(value1)
                                        performance_values.append(value1)

                                    variable_to_append.append(["lower_red_1_2:"])
                                    if isinstance(value2, list):
                                        variable_to_append.append(value2[0])
                                        performance_values.append(value2[0])
                                    else:
                                        variable_to_append.append(value2)
                                        performance_values.append(value2)

                                    variable_to_append.append(["lower_red_1_3:"])
                                    if isinstance(value3, list):
                                        variable_to_append.append(value3[0])
                                        performance_values.append(value3[0])
                                    else:
                                        variable_to_append.append(value3)
                                        performance_values.append(value3)

                                    variable_to_append.append(["upper_red_1_1:"])
                                    if isinstance(value4, list):
                                        variable_to_append.append(value4[0])
                                        performance_values.append(value4[0])
                                    else:
                                        variable_to_append.append(value4)
                                        performance_values.append(value4)


                                    variable_to_append.append(["upper_red_1_2:"])
                                    if isinstance(value3, list):
                                        variable_to_append.append(value5[0])
                                        performance_values.append(value5[0])
                                    else:
                                        variable_to_append.append(value5)
                                        performance_values.append(value5)

                                    variable_to_append.append(["upper_red_1_3:"])
                                    if isinstance(value3, list):
                                        variable_to_append.append(value6[0])
                                        performance_values.append(value6[0])
                                    else:
                                        variable_to_append.append(value6)
                                        performance_values.append(value6)
                                    finmask = Perform_Extinguisher_Net(image,performance_values)

                                    used_variables.append(variable_to_append)
                                    mask_variations.append(finmask)
                                    cv2.imwrite(os.path.join("D:/Programmieren/MasterOfDisaster/Experiments_and_Implementations/Extinguisher_Net/images",address),  finmask)

    """
    for immask in mask_variations:
        plt.imshow(immask)
        plt.show()
        plt.imshow(image)
        plt.show()
    """
    return  used_variables, mask_variations