import csv
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL.Image import Resampling
from skimage.filters import frangi
from skimage.morphology import remove_small_objects
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw  # Import ImageDraw from PIL
from skimage import morphology, measure

from statistical_calc import binomial_distribution

######################################################################################################################################################################################
######################################################################################################################################################################################
BRANCHLENGTH_THRESHOLD = 5
NUMBER_OF_PHOTOS = 4
global BRLengths
global actual_image
global segment_image
global skele_image
global DD

BRLengths = []
Logs = []
CC=[]
DD = []

######################################################################################################################################################################################
def load_image():
    global input_image, file_path, actual_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
    if file_path:
        input_image = Image.open(file_path)
        photo = calculate_image_size(image=input_image)
        left_frame.image_label.config(image=photo)
        left_frame.image_label.image = photo
        print(f"Loaded image: {file_path}")
        actual_image = input_image


######################################################################################################################################################################################
def create_segmentation():
    global input_image, file_path, contours2, line_coordinates, contours, branch_points, num_objectsCompound, num_objects2, skele_image, segment_image, slider_value3, asli_contours

    # file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

    if file_path:
        # threshold = slider1_var.get()
        # binary_image = input_image.point(lambda p: p > threshold and 255)

        I = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # I=input_image
        Ip = I.astype(float)
        thr = np.percentile(Ip[Ip > 0], 1) * 0.9
        Ip[Ip <= thr] = thr
        Ip = Ip - np.min(Ip)
        Ip = Ip / np.max(Ip)

        input_image = Ip
        # Create a range of sigma values from 0.5 to 3 with an increment of 0.5
        sigmas_range = np.arange(0.25, 3.5, 0.25)

        # Convert the input image to grayscale
        gray_image = input_image

        # Create a list to store the filtered images
        filtered_images = []

        # Loop through the sigma values
        for sigma in sigmas_range:
            # Apply the Frangi filter for the current sigma
            filtered_image = frangi(
                gray_image,
                sigmas=(sigma, sigma),  # Use the current sigma for both dimensions
                scale_step=2,
                black_ridges=False
            )

        filtered_images.append(filtered_image)

        # Create a combined image by taking the maximum value across all filtered images
        combined_image = np.max(filtered_images, axis=0)
        filtered_image = combined_image
        # Convert the filtered image to a black and white image
        # You can adjust the threshold value as needed
        # threshold = 1e-5
        # threshold = slider1_var.get()
        slider_value = slider1_var.get()  # Replace with the actual function to get the slider value

        # Define a mapping of slider values to threshold values
        threshold_mapping = {
            1: 1e-4,
            2: 1e-5,
            3: 1e-6,
            4: 1e-7,
            5: 1e-8,
            6: 1e-9,
            7: 1e-10,
            8: 1e-11,
            9: 1e-12,
            10: 1e-13
        }

        # Check if the slider value is in the mapping
        if slider_value in threshold_mapping:
            threshold = threshold_mapping[slider_value]
        else:
            # Default threshold value for other cases
            threshold = 1e-5  # You can set a default value here if needed

        # Now 'threshold' contains the threshold value based on the slider value
        ########################################################################print(f"Threshold value: {threshold}")
        bw_image = filtered_image > threshold

        # Remove small objects (noise) from the binary image
        min_size = 200  # Adjust this size threshold as needed
        cleaned_image = remove_small_objects(bw_image, min_size=min_size)
        cleaned_image2 = (cleaned_image * 255).astype(np.uint8)

        pil_image2 = Image.fromarray(cleaned_image2)
        # Create a PhotoImage from the PIL Image
        photo2 = calculate_image_size(image=pil_image2)
        # Update the new frame (second input image frame)
        second_input_frame.image_label.config(image=photo2)
        second_input_frame.image_label.image = photo2
        segment_image = pil_image2

        # Binarize and skeletonize the input image
        bwskel_image = cv2.ximgproc.thinning(cleaned_image2)

        labeled_image, num_objectsCompound = measure.label(bwskel_image, connectivity=2, return_num=True)
        ########################################################################print("Compound branches detected at start=========================="+ str(num_objectsCompound))

        pil_image = Image.fromarray(bwskel_image.astype(np.uint8))
        photo = calculate_image_size(image=pil_image)
        # plt.plot(256, 256, '.g')

        right_frame.image_label.config(image=photo)
        right_frame.image_label.image = photo
        ########################################################################print(f"Segmentation created with threshold: {threshold}")

        # Convert the NumPy array to a PIL Image
        # Find contours in the binary image
        pil_image = Image.fromarray(bwskel_image.astype(np.uint8))
        pil_image_with_line = pil_image.copy()
        pil_image_with_line = pil_image_with_line.convert('RGB')

        # Create a draw object on the copy of the PIL Image
        draw = ImageDraw.Draw(pil_image_with_line)
        # Get the dimensions of the image
        image_width, image_height = pil_image_with_line.size
        # Calculate the center coordinates
        center_x = image_width // 2
        center_y = image_height // 2
        # Define the line coordinates (from top to bottom)
        line_coordinates = [(0, center_x), (image_width, center_y)]
        # Draw the line on the copy of the PIL Image
        draw.line(line_coordinates, fill="green", width=2)
        # Update the displayed image in the right frame
        updated_photo = calculate_image_size(image=pil_image_with_line)
        right_frame.image_label.config(image=updated_photo)
        right_frame.image_label.image = updated_photo
        ########################################################################print("Base Line Line drawn")

        contours, _ = cv2.findContours(bwskel_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a copy of the PIL Image for drawing contours
        contour_image = pil_image_with_line.copy()
        print("Compound Branch Contours..................>" + str(len(contours)))
        asli_contours = contours

        # Create a draw object on the contour image
        draw = ImageDraw.Draw(contour_image)

        # Loop through each contour and draw it in green
        for contour in contours:
            for point in contour:
                x, y = point[0]  # Get the coordinates of each point in the contour
                draw.point((x, y), fill="red")

        # Convert the contour image to PhotoImage
        contour_photo = calculate_image_size(image=contour_image)

        # Update the displayed image in the right frame
        right_frame.image_label.config(image=contour_photo)
        right_frame.image_label.image = contour_photo

        Ex_right_frame.image_label.config(image=contour_photo)
        Ex_right_frame.image_label.image = contour_photo

        ########################################################################print("Contours drawn")
        skele_image = contour_image

        # Load the image
        I = cleaned_image2
        # Apply binary thresholding
        _, binary_image = cv2.threshold(I, 128, 1, cv2.THRESH_BINARY)
        # Perform skeletonization
        skeleton_img = morphology.skeletonize(binary_image)
        # Label connected components
        labeled_image, num_labels = measure.label(skeleton_img, connectivity=1, return_num=True)
        # Initialize a list to store branch points
        branch_points = []
        # Iterate through labeled objects and find branch points
        for label in range(1, num_labels + 1):
            object_pixels = np.argwhere(labeled_image == label)
            neighbors = {}  # Store neighboring pixels for each object
            for pixel in object_pixels:
                x, y = pixel[0], pixel[1]
                neighbor_count = np.sum(labeled_image[x - 1:x + 2, y - 1:y + 2] == label) - 1
                if neighbor_count > 2:
                    branch_points.append((x, y))


        # Initialize a counter for the green ellipses
        green_ellipse_count = 0

        # Create a copy of the right frame image
        # Overlay branch points on the right frame image in blue
        for point in branch_points:
            x, y = point
            draw.point((y, x), fill=(0, 0, 255))  # Blue color
            point_radius = 5
            left_x = max(0, y - point_radius)
            upper_y = max(0, x - point_radius)
            right_x = min(image_width - 1, y + point_radius)
            lower_y = min(image_height - 1, x + point_radius)
            draw.ellipse((left_x, upper_y, right_x, lower_y), fill=(0, 255, 0))

            # Increment the counter
            green_ellipse_count += 1

        # Print the number of green ellipses
        #print(f'Number of green ellipses: {green_ellipse_count}')

        # Convert the right frame image with branch points to PhotoImage
        right_frame_photo = calculate_image_size(image=contour_image)
        # Update the displayed image in the right frame
        right_frame.image_label.config(image=right_frame_photo)
        right_frame.image_label.image = right_frame_photo
        ########################################################################print("Branch points overlayed")

        skeleton_img = skeleton_img.astype(np.uint8)
        for point in branch_points:
            x, y = point
            skeleton_img[x, y] = 0

        labeled_image2, num_objects2 = measure.label(skeleton_img, connectivity=2, return_num=True)
        print("Detached branches after removing branch points ===================" + str(num_objects2))

        slider_value2 = slider2_var.get()  # Replace with the actual function to get the slider value

        # Define a mapping of slider values to threshold values
        threshold_mapping2 = {
            1: 5,
            2: 20,
            3: 35,
            4: 50,
            5: 70,
            6: 90,
            7: 120,
            8: 150,
            9: 180,
            10: 200
        }

        # Check if the slider value is in the mapping
        if slider_value2 in threshold_mapping2:
            BRANCHLENGTH_THRESHOLD = threshold_mapping2[slider_value2]
        else:
            # Default threshold value for other cases
            BRANCHLENGTH_THRESHOLD = 50  # You can set a default value here if needed

        skeleton_img = skeleton_img.astype(np.uint8)
        contours2, _ = cv2.findContours(skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = [contour for contour in contours2 if len(contour) >= BRANCHLENGTH_THRESHOLD]

        # Create a draw object on the contour image
        draw = ImageDraw.Draw(contour_image)

        # Loop through each contour and draw it in green
        for contour in contours2:
            for point in contour:
                x, y = point[0]  # Get the coordinates of each point in the contour
                draw.point((x, y), fill="blue")
                # Define the radius of the circle
                circle_radius = 2  # Adjust the size as needed

                x1 = x - circle_radius
                y1 = y - circle_radius
                x2 = x + circle_radius
                y2 = y + circle_radius

                # Draw a red circle around the point
                draw.ellipse([x1, y1, x2, y2], outline="blue", width=2)

        # Convert the contour image to PhotoImage
        contour_photo = calculate_image_size(image=contour_image)

        Ex_right_frame.image_label.config(image=contour_photo)
        Ex_right_frame.image_label.image = contour_photo

        return contours2


######################################################################################################################################################################################
def find_unique_rows_in_contours(contours2):
    unique_contours = []

    for curve in contours2:
        # Convert the curve to a NumPy array
        curve_array = np.array(curve)
        # Find unique rows for the current curve
        unique_rows = np.unique(curve_array, axis=0)
        # Append the unique rows for this curve to the result list
        unique_contours.append(unique_rows)

    return unique_contours


######################################################################################################################################################################################

def create_segmentation2():
    global CC
    CC.clear()
    DD.clear()
    Logs.clear()
    BRLengths.clear()


    CC = create_segmentation()
    # Call the function to find unique rows in contours2
    # unique_contours = find_unique_rows_in_contours(CC)
    print("Retained branches after some/non-dendrites using Branch Length THRESHOLD================" + str(len(CC)))


######################################################################################################################################################################################
def save_Framed_images():
    folder_path, file_name_with_extension = os.path.split(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name_with_extension)

    # Save the original image
    new_file_savenameOI = file_name_without_extension + "_OriginalImage.png"
    actual_image.save(os.path.join(folder_path, new_file_savenameOI))

    # Save the segmented image
    new_file_savenameSI = file_name_without_extension + "_BW_Segmentation.png"
    segment_image.save(os.path.join(folder_path, new_file_savenameSI))

    # Save the skeletonized image
    new_file_savenameBWS = file_name_without_extension + "_BW_Skeleton.png"
    skele_image.save(os.path.join(folder_path, new_file_savenameBWS))

    print(" ALL Images saved in Base Directory ")


######################################################################################################################################################################################    
# Define a custom function to find the nearest point
def find_nearest_point(point, curve1):
    distances = np.sqrt(np.sum((curve1 - point) ** 2, axis=1))
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    return curve1[min_index], min_index, min_distance

######################################################################################################################################################################################
def generate_results():
    unique_contours = find_unique_rows_in_contours(CC)
    plot_contours(CC)

    i = 0
    # fig, ax = plt.subplots()
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    # plt.plot([x1, x2], [y1, y2])
    for thiscontour in unique_contours:
        XY = thiscontour[:, 0, :]
        # Plot the base line on the same image
        # Overlay the directional contour
        # plt.plot(XY[:, 0], XY[:, 1], '.', color=np.random.rand(3))
        # Fit a linear polynomial to the contour
        p = np.polyfit(XY[:, 0], XY[:, 1], 1)
        # Evaluate the fitted polynomial
        f = np.polyval(p, XY[:, 0])
        XY[:, 1] = f
        ttt = np.unique(XY, axis=0)
        DD.append(ttt)
        # Plot the fitted line
        # plt.plot(XY[:, 0], f, '-k')
        # Calculate the angle between the fitted line and the base line
        # dir_vec1 = np.array([511 - 0, 0 - 0])
        # dir_vec2 = np.array([XY[-1, 0] - XY[0, 0], XY[-1, 1] - XY[0, 1]])
        # dot_product = np.dot(dir_vec1, dir_vec2)
        # magnitude1 = np.linalg.norm(dir_vec1)
        # magnitude2 = np.linalg.norm(dir_vec2)
        # cosine_theta = dot_product / (magnitude1 * magnitude2)
        # angle_degrees = np.degrees(np.arccos(cosine_theta))
        ## Display the angle
        # print(f'Angle between Base and Line '+str(i)+ '  is '+str(angle_degrees)+' degrees')
        # Angles.append(angle_degrees)
    # ax.invert_yaxis()
    # plt.show()

    fig = plt.figure("Preview Measured Branch Estimation on Original Image")
    plt.imshow(actual_image)
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    plt.plot([x1, x2], [y1, y2])
    for thiscontour in unique_contours:
        XY = thiscontour[:, 0, :]
        i = i + 1
        # Plot the base line on the same image
        # Overlay the directional contour
        # plt.plot(XY[:, 0], XY[:, 1], '.', color=np.random.rand(3))
        # Fit a linear polynomial to the contour
        p = np.polyfit(XY[:, 0], XY[:, 1], 1)
        # Evaluate the fitted polynomial
        f = np.polyval(p, XY[:, 0])
        XY[:, 1] = f
        ttt = np.unique(XY, axis=0)
        # Plot the fitted line
        plt.plot(XY[:, 0], f, '-y', linewidth=3)
        plt.title(" Original Image with overlaid detected Branches")
    plt.show()

    ######################################################################################################################################################################################

    # Initialize variables

    for ii in range(len(DD)):
        curve1 = DD[ii]
        # Get the first point
        p1x, p1y = curve1[0, 0], curve1[0, 1]
        # Get the last point
        p2x, p2y = curve1[-1, 0], curve1[-1, 1]
        # Calculate the Euclidean distance between the first and last points
        BranchLen = np.sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)
        # Print the distance
        #print(f"Length of Branch {ii} is {BranchLen:.2f}")
        BRLengths.append(BranchLen)

        # plt.plot(curve1[:, 0], curve1[:, 1], '.r')

        for jj in range(len(DD)):
            if jj != ii:
                curve2 = DD[jj]
                curve2 = np.array(curve2)  # Ensure curve2 is a NumPy array
                curve2_resampled = np.zeros_like(curve1)

                # Resample curve2 to match the length of curve1 (adjust as needed)
                for i in range(len(curve1)):
                    idx = int(i * len(curve2) / len(curve1))
                    curve2_resampled[i] = curve2[idx]

                # plt.plot(curve2_resampled[:, 0], curve2_resampled[:, 1], '.g')

                # Initialize variables to store the closest point and its distance
                closest_point = None
                Mins = []

                for i in range(len(curve2_resampled)):
                    point2 = np.abs(curve2_resampled[i])
                    Point, PointID, Min = find_nearest_point(point2, curve1)
                    Mins.append(Min)

                v = min(Mins)
                id = np.argmin(Mins)
                closest_point = curve2_resampled[id]

                # plt.plot(closest_point[0], closest_point[1], '.r', markersize=20)
                c1, c2 = ii, jj
                cx, cy = closest_point[0], closest_point[1]
                dst = v
                CRow = [c1, c2, cx, cy, dst]
            else:
                c1, c2, cx, cy, dst = ii, jj, 0, 0, 99999
                CRow = [c1, c2, cx, cy, dst]

            Logs.append(CRow)

    # plt.show()

    # Assuming 'Logs' is a list of lists with 5 columns: [curve1ID, curve2ID, x, y, distance]
    # it can adapt this code according to actual data structure.

    # Create a dictionary to store the closest curve and its distance for each curve in column 1
    closest_curves = {}

    # Iterate through the 'Logs' data
    for row in Logs:
        curve1ID, curve2ID, x, y, distance = row

        # If curve1ID is not in the dictionary or the new distance is shorter, update the closest curve
        if curve1ID not in closest_curves or distance < closest_curves[curve1ID][1]:
            closest_curves[curve1ID] = (curve2ID, distance)

    # Print the closest curve for each curve in column 1
    '''for curve1ID, (closest_curve2ID, shortest_distance) in closest_curves.items():
        print(f"Branch {curve1ID} is closest to Branch {closest_curve2ID} with a distance of {shortest_distance}")'''

    return Logs


######################################################################################################################################################################################
######################################################################################################################################################################################

def analyze_and_plot_parallel_branches(contours, angle_threshold):
    def compute_slope(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x2 - x1 == 0:
            return float('inf')  # Vertical line (infinite slope)
        return (y2 - y1) / (x2 - x1)

    def are_lines_parallel(slope1, slope2, parallel_threshold=0.1):
        return abs(slope1 - slope2) < parallel_threshold

    def group_parallel_branches(contours, angle_threshold):
        global angles
        global gbcount
        global branch_info_list
        grouped_branches = {}
        angles = []
        gbcount = []
        branch_info_list = []  # Initialize a list to store branch information

        for contour in contours:
            first_point = contour[0]
            last_point = contour[-1]
            slope = compute_slope(first_point, last_point)
            angle = np.arctan(slope) * 180 / np.pi
            if angle < 0:
                angle = 360 + angle
            angles.append(angle)
            # Create a dictionary for the branch information
            branch_info = {
                'First Point': first_point,
                'Last Point': last_point
            }
            # Append the branch information dictionary to the list
            branch_info_list.append(branch_info)

        print("\nsum all lines: ", len(angles))
        #print(len(contours))
        print("Print the values of Angles:" ,angles)
        print("Print the values of Branches Length:", BRLengths,"\n")



        # Dictionary to store cumulative lengths for each branch
        cumulative_lengths = {}

        for i, contour in enumerate(contours):
            angle1 = angles[i]  # Get the angle of the current branch
            if len(contour) > 1:
                cumulative_length = 0
                for j in range(len(contour) - 1):
                    x1, y1 = contour[j]
                    x2, y2 = contour[j + 1]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    cumulative_length += distance

                cumulative_lengths[i] = cumulative_length


            group = [i]
            for j, angle2 in enumerate(angles):
                if i != j and abs(angle1 - angle2) <= angle_threshold:
                    group.append(j)
            grouped_branches[i] = group

        # Sort grouped_branches by the number of items in each group (descending order)
        sorted_branches = dict(sorted(grouped_branches.items(), key=lambda item: len(item[1]), reverse=True))


        # Iterate over sorted branches
        for key, group in sorted_branches.items():
            # Iterate over branches in the current group
            for branch in group:
                # Remove the branch from other groups
                for other_key, other_group in sorted_branches.items():
                    if other_key != key and branch in other_group:
                        other_group.remove(branch)
        sorted_branches_parallel= {}
        for key, group in sorted_branches.items():
          if len(group)>0 :
              sorted_branches_parallel[key] = group

        sorted_branches = dict(sorted(sorted_branches_parallel.items(), key=lambda item: len(item[1]), reverse=True))
        print(sorted_branches)

        return sorted_branches, cumulative_lengths

    # Sort and deduplicate branches in parallel groups
    group_sort_branch, cumulative_lengths = group_parallel_branches(contours, angle_threshold=angle_threshold)

    # Count the number of branches in each sorted parallel group
    sorted_group_branch_counts = {group: len(branches) for group, branches in group_sort_branch.items()}

    # Count the total number of sorted parallel groups
    num_sorted_parallel_groups = len(group_sort_branch)

    print(f"-------------------------Sorted Parallel Group information :---------------------------------:")
    # Print sorted parallel group information, including angles, branch coordinates, and branch counts

    parallel_list_branch = []
    parallel_list_all_branch = []  # Initialize a list to store branch information
    for group, branches in group_sort_branch.items():
        num_parallel_lines = sorted_group_branch_counts.get(group, 0)
        if len(branches) > 1:
            parallel_list_branch.append(branches)
            parallel_list_all_branch.append(branches)

        # Skip groups with zero parallel lines
        if num_parallel_lines == 0:
            continue

        group_info = {
            'Number of Parallel Lines': sorted_group_branch_counts[group],
            'Branches': []
        }

        print(f"\nNumber of Parallel Lines in this Group --------: {sorted_group_branch_counts[group]}")
        thisgcount = sorted_group_branch_counts[group]
        gbcount.append(thisgcount)
        for branch in branches:
            print(f"  Key {branch}:")
            branch_coordinates = contours[branch]
            first_point = branch_coordinates[0]
            last_point = branch_coordinates[-1]
            slope = compute_slope(first_point, last_point)
            angle = np.arctan(slope) * 180 / np.pi
            if angle < 0:
                angle = 360 + angle

            print(f"    Angle: {angle:.2f} degrees" + f" First Point: {first_point}" + f" Last Point: {last_point}")

            # Print the cumulative length for the branch
            if branch in cumulative_lengths:
                length = cumulative_lengths[branch]
                print(f"    Branch Length: {length:.2f} units")

                branch_info = {
                    'Key': branch,
                    'Angle': angle,
                    'First Point': first_point,
                    'Last Point': last_point,
                    'Length': length
                }
                # Append the branch information dictionary to the group_info dictionary
                group_info['Branches'].append(branch_info)
        # Append the group_info dictionary to the parallel_list_branch list
        parallel_list_all_branch.append(group_info)
    print(parallel_list_all_branch)

    # Optionally, visualize the branches as well
    plt.figure('Display 2D marked lines based on Image Segmentation')
    for i, contour in enumerate(contours):
        x, y = contour[:, 0], contour[:, 1]
        # plt.plot(x, y, label=f"Branch {i}")
        plt.plot(x, y)
        plt.title("2D marked lines based on Image Segmentation")
        # Add text annotation to the left of the graph
        plt.text(x[0], y[0], f"Branch {i}", verticalalignment='center', fontsize=8)


    return angles, parallel_list_branch, gbcount, contours, parallel_list_all_branch



######################################################################################################################################################################################
global ANG_TH


######################################################################################################################################################################################
def update_slider_value1(slider_var, label):
    slider_value = slider_var.get()
    # label.config(text=f"Slider Value: {slider_value}")
    # print(f"tHRESHOLD: {slider_value}")


######################################################################################################################################################################################
def update_slider_value2(slider_var, label):
    slider_value = slider_var.get()
    # label.config(text=f"Slider Value: {slider_value}")
    # print(f"DISTANCE: {slider_value}")


######################################################################################################################################################################################
def update_slider_value3(slider_var, label):
    slider_value = slider_var.get()
    # label.config(text=f"Slider Value: {slider_value}")
    # print(f"ANGLE: {slider_value}")


######################################################################################################################################################################################

def write_results():
    """slider_value3 = slider3_var.get()  # Replace with the actual function to get the slider value
    # Define a mapping of slider values to threshold values
    threshold_mapping3 = {
        1: 1,
        2: 2,
        3: 5,
        4: 10,
        5: 15,
        6: 20,
        7: 25,
        8: 30,
        9: 35,
        10: 40
    }"""

    """# Check if the slider value is in the mapping
    if slider_value3 in threshold_mapping3:
        ANG_TH = threshold_mapping3[slider_value3]
    else:
        # Default threshold value for other cases
        ANG_TH = 10  # can set a default value here if needed"""
    ANG_TH = 5 if 180 / len(DD) < 36 else (180 / len(DD))
    print("selected ANGLE for parallel detecting::", ANG_TH)

    # Angles, num_groups, gbcount,Mcontours=analyze_and_plot_parallel_branches(DD, angle_threshold=ANG_TH)
    Angles, parallel_list_branch, gbcount, Mcontours, parallel_list_all_branch = analyze_and_plot_parallel_branches(DD, angle_threshold=ANG_TH)
    print(parallel_list_branch)
    fig = plt.figure("Measured Branch Estimation on Original Image")
    plt.imshow(actual_image)
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    plt.plot([x1, x2], [y1, y2])
    # Optionally, visualize the branches as well
    for i, contour in enumerate(Mcontours):
        x, y = contour[:, 0], contour[:, 1]
        plt.plot(x, y, '-y', linewidth=3)
        plt.title(" Measured Branch Estimation Plotted over 2D image")

    plt.show()

    # Count the frequency of each unique value
    value_counts = {}
    gbcount_sorted = sorted(gbcount)
    for value in gbcount_sorted:
        if value != 1:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
    sorted(value_counts)

    # Extract unique values and their frequencies and sort them
    unique_values = sorted(list(value_counts.keys()))
    frequencies = [value_counts[value] for value in unique_values]
    plt.figure("Measured Classification of dendritic branch parallel growth ")
    # Create a bar graph
    plt.bar(range(len(unique_values)), frequencies)
    plt.xticks(range(len(unique_values)), unique_values)

    # Add labels and title
    plt.xlabel('Parallel Groups with Branches')
    plt.ylabel('Frequency Count for respective groups')
    plt.title("DETECTED-Total branches== " + str(sum(gbcount)) + ", Arranged into" + str(sum(frequencies)) + " groups")
    plt.show()

    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    print(
        "--------------------------- AFTER MEASURED BRANCH ANALYSIS, NOW BELOW IS THE INFORMATION FOR SIMULATED RANDOM LINES----------------------------------------")

    # Parameters for the binomial distribution
    num_lines = sum(gbcount)
    gbcountRANDOM= binomial_distribution(num_lines)
    #AnglesRANDOM, num_groupsRANDOM, gbcountRANDOM, contoursRANDOM = analyze_and_plot_parallel_branches(BINOMIAL_LINES,angle_threshold=ANG_TH)
    """ 
    fig = plt.figure()
    plt.imshow(actual_image)
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    plt.plot([x1, x2], [y1, y2])
    # Optionally, visualize the branches as well
    for i, contour in enumerate(contoursRANDOM):
        x, y = contour[:, 0], contour[:, 1]
        plt.plot(x, y, '-y', linewidth=3)
        plt.title(" Random Branch Estimation Plotted over 2D image")

    #plt.show()
    """

    value_countsR = {}
    for valueR in range(2, len(gbcountRANDOM)):
            value_countsR[valueR] = gbcountRANDOM[valueR]

    # Extract unique values and their frequencies and sort them
    unique_valuesR = sorted(list(value_countsR.keys()))
    frequenciesR = [value_countsR[valueR] for valueR in unique_valuesR]
    plt.figure("Simulation Classification of dendritic branch parallel growth ")
    # Create a bar graph
    plt.bar(range(len(unique_valuesR)), frequenciesR)
    plt.xticks(range(len(unique_valuesR)), unique_valuesR)

    # Add labels and title
    plt.xlabel('Parallel Groups with Branches')
    plt.ylabel('Frequency Count for respective groups')
    plt.title(
        "RANDOM-Total branches== " + str(num_lines) + ", Arranged into" + str(sum(frequenciesR)) + " groups")
    plt.show()

    # Count the frequency of each unique value
    value_counts = {}
    gbcount_sorted = sorted(gbcount)
    for value in gbcount_sorted:
        if value!=1:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
    sorted(value_counts)
    # Print the value-frequency pairs STATISTICS for Measured Classification
    measur_Rparalle_data_list= []
    print("STATISTICS for Measured Classification")
    print("value : frequency")
    for value, frequency in value_counts.items():
        print(f"{value}     :  {frequency}")
        measur_Rparalle_data_list.append((value, frequency))

    stat_Rparalle_data_list= []
    # Print the value-frequency pairs - STATISTICS for Random Classification
    print("STATISTICS for Random Classification")
    print("value : frequency")
    for i in range(2, len(gbcountRANDOM)):
        print(f'{i}     :  {gbcountRANDOM[i]}')
        stat_Rparalle_data_list.append((i, gbcountRANDOM[i]))

    PercentagePS, PercentagePM, PercentagePMS, LongtermES, sum_all_measured_lines, sum_all_simulation_lines = disp_final_stats(num_lines, value_counts, gbcountRANDOM)

    """# Extract relevant data
    filtered_branches = []
    for branch in parallel_list_all_branch[1]:
        if branch.get('Number of Parallel Lines', 0) < 2:
            filtered_branches.append(branch)

    # Extract lengths and angles
    lengths = [branch['Length'] for branch in filtered_branches]
    angles = [branch['Angle'] for branch in filtered_branches]

    # Calculate average and standard deviation
    average_length = np.mean(lengths)
    std_dev_length = np.std(lengths)

    average_angle = np.mean(angles)
    std_dev_angle = np.std(angles)

    # Print results
    print(f"\nAverage Length: {average_length:.2f}")
    print(f"Standard Deviation of Length: {std_dev_length:.2f}")

    print(f"\nAverage Angle: {average_angle:.2f}")
    print(f"Standard Deviation of Angle: {std_dev_angle:.2f}")"""


    ######################################################################################################################################################################################

    # Extract the folder path from the file path
    folder_path, file_name_with_extension = os.path.split(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name_with_extension)
    new_file_name = file_name_without_extension + ".csv"
    csv_file_path = os.path.join(folder_path, new_file_name)

    #BranchesCompound_df = pd.DataFrame({num_objectsCompound})
    #BranchesDetached_df = pd.DataFrame({num_objects2})
    # BP_df = pd.DataFrame(branch_points)

    Logs_df = pd.DataFrame(Logs)
    BL_df = pd.DataFrame(BRLengths)
    ANGLE_df = pd.DataFrame(Angles)
    BP_df = pd.DataFrame(branch_info_list)
    BranchesRetained_df = pd.DataFrame({len(DD)})
    sumPgroupMeasured_df = pd.DataFrame({sum(frequencies)})
    sum_all_measured_lines_df = pd.DataFrame({sum_all_measured_lines})
    sumPgroupSimulation_df = pd.DataFrame({sum(frequenciesR)})
    sum_all_simulation_lines_df = pd.DataFrame({sum_all_simulation_lines})
    PercentagePS_df = pd.DataFrame({PercentagePS})
    PercentagePM_df = pd.DataFrame({PercentagePM})
    PercentagePMS_df = pd.DataFrame({PercentagePMS})
    LongtermES_df = pd.DataFrame({LongtermES})
    parallel_list_branch_df = pd.DataFrame({'Parallels Groups list': [','.join(map(str, inner_list)) for inner_list in parallel_list_branch]})
    measur_Rparalle_data_list_df = pd.DataFrame(measur_Rparalle_data_list, columns=['VALUE', 'FREQUENCY'])
    stat_Rparalle_data_list_df = pd.DataFrame(stat_Rparalle_data_list, columns=['VALUE', 'FREQUENCY'])
    FileName_df = pd.DataFrame({file_path})

    result_df = pd.concat(
        [Logs_df, BL_df, ANGLE_df, BP_df, BranchesRetained_df, sumPgroupMeasured_df,sum_all_measured_lines_df,
         sumPgroupSimulation_df ,sum_all_simulation_lines_df,
         PercentagePS_df, PercentagePM_df, PercentagePMS_df, LongtermES_df,parallel_list_branch_df,measur_Rparalle_data_list_df, stat_Rparalle_data_list_df, FileName_df],
        axis=1, ignore_index=True)

    # Create a sample DataFrame
    result_df2 = pd.DataFrame({'Current Branch': result_df[0][:],
                               'Relative Branch': result_df[1][:],
                               'Closest X': result_df[2][:],
                               'Closest Y': result_df[3][:],
                               'Shortest Distance[µm]': result_df[4][:],
                               'Branch Length[µm]': result_df[5][:],
                               'Branch Angle[°]': result_df[6][:],
                               'Branch First Point': result_df[7][:],
                               'Branch Last Point': result_df[8][:],
                               'Detached Branches': result_df[9][:],
                               'Measured-Number of Parallels Groups': result_df[10][:],
                               'Measured- Number of Parallels Lines': result_df[11][:],
                               'Simulation- Number of Parallels Groups': result_df[12][:],
                               'Simulation- Number of Parallels Lines': result_df[13][:],
                               'Percentage of simulation parallelism': result_df[14][:],
                               'Measurement Parallelism Percentage': result_df[15][:],
                               'Percentage of Parallelism of Measurement Relative to Simulation': result_df[16][:],
                               'Weights Ratio in the Long-term Of (E)\(S)': result_df[17][:],
                               'Parallels Groups List': result_df[18][:],
                               'Measured VALUE': result_df[19][:],
                               'Measured FREQUENCY ': result_df[20][:],
                               'Simulation VALUE': result_df[21][:],
                               'Simulation FREQUENCY ': result_df[22][:],
                               'Image File Processed': result_df[23][:],
                               })

    # Save the DataFrame to a CSV file
    result_df2.to_csv(csv_file_path, index=False)
    print(f'\nData has been saved to {csv_file_path}')

    save_Framed_images()
    create_graphs(BRLengths, Angles)

    # Input file path (change this to your actual file path)
    # Get folder path, file name, and extension
    folder_path, file_name_with_extension = os.path.split(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name_with_extension)

    # Create a new CSV file name with the same name as the original file
    new_file_name2 = file_name_without_extension +" Dendrites Groups" + ".csv"

    # Create the full path for the new CSV file
    csv_file_path2 = os.path.join(folder_path, new_file_name2)

    # Open the CSV file for writing
    with open(csv_file_path2, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header
        csv_writer.writerow(["Set", "Branch Key", "Angle[°]", "First Point", "Last Point", "Length[µm]"])

        # Iterate through the data and write to CSV
        for i in range(0, len(parallel_list_all_branch), 2):
            # Check if there are enough elements in data_list
            if i + 1 < len(parallel_list_all_branch):
                keys = parallel_list_all_branch[i]
                parallel_data = parallel_list_all_branch[i + 1]
                branches = parallel_data['Branches']

                for branch in branches:
                    key = branch['Key']
                    angle = "{:.2f}".format(branch['Angle'])
                    first_point = branch['First Point']
                    last_point = branch['Last Point']
                    length = "{:.2f}".format(branch['Length'])
                    csv_writer.writerow([i // 2 + 1, key, angle, first_point, last_point, length])




######################################################################################################################################################################################

def create_graphs(BRLengths, Angles):
    plt.figure("Length distribution of of dendritic branches")
    # Create a bar graph for branch lengths
    plt.bar(range(len(BRLengths)), BRLengths)
    # Add labels and a title
    plt.xlabel(' Branches')
    plt.ylabel('Length in Unit Pixels [µm]')
    plt.title('Branch Length Distribution')
    # Calculate and display average and standard deviation for branch lengths
    avg_length = np.mean(BRLengths)
    std_dev_length = np.std(BRLengths)
    plt.text(len(BRLengths), max(BRLengths) , f'Avg: {avg_length:.2f}\nStd Dev: {std_dev_length:.2f}',color='black',
             fontsize=12, bbox=dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5), horizontalalignment='right')
    # Show the graph
    plt.show()

    # Create a bar graph for branch angles
    plt.figure("Angular distribution of dendritic branches")
    plt.bar(range(len(Angles)), Angles)
    # Add labels and a title
    plt.xlabel(' Branches')
    plt.ylabel('Angle in Degrees with Horizontal Base[°]')
    plt.title('Angular Distribution')
    # Calculate and display average and standard deviation for branch angles
    avg_angle = np.mean(Angles)
    std_dev_angle = np.std(Angles)
    plt.text(len(Angles) , max(Angles) , f'Avg: {avg_angle:.2f}°\nStd Dev: {std_dev_angle:.2f}°', color='black',
             fontsize=12, bbox=dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5), horizontalalignment='right')
    # Show the graph
    plt.show()


######################################################################################################################################################################################
from collections import Counter


def plot_contours(CC):
    # Create a figure and axis
    fig, ax = plt.subplots()
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    plt.plot([x1, x2], [y1, y2], label='Base')

    # Loop through the contours and plot each one with a label
    for i, contour in enumerate(CC):
        # Convert the contour to a NumPy array
        contour = np.array(contour)
        XY = contour[:, 0]
        # Extract the x and y coordinates from the contour
        x = XY[:, 0]
        y = XY[:, 1]
        # Plot the contour with a label
        plt.plot(x, y)

    # Add a legend to the plot
    plt.legend()
    ax.invert_yaxis()
    plt.title(" 2D Marked Branches based on Image Segmentation")
    fig.canvas.manager.set_window_title('Preview Display 2D Marked lines based on Image Segmentation')
    plt.show()

######################################################################################################################################################################################
def disp_final_stats(num_lines, value_counts, gbcountRANDOM):
    all_lines_detected_in_original_image = num_lines

    print('\n',
          "<--------------- Calculation of the percentage of parallelism in relation to the simulation: --------------->",
          '\n')
    # simulation
    sum_all_simulation_lines = 0
    for i in range(2, len(gbcountRANDOM)):
        sum_all_simulation_lines += i * gbcountRANDOM[i]
    PercentagePS = sum_all_simulation_lines / all_lines_detected_in_original_image
    print("simulation: ", PercentagePS)

    # measured
    sum_all_measured_lines = 0
    for value, frequency in value_counts.items():
        sum_all_measured_lines += value * frequency
    PercentagePM = sum_all_measured_lines / all_lines_detected_in_original_image
    print("measured: ", PercentagePM)

    PercentagePMS = sum_all_measured_lines / sum_all_simulation_lines
    print("E\S: ", PercentagePMS)

    print('\n', "<---------------  Long - term  parallels of E\S: --------------->", '\n')
    # simulation with weights - in the long term
    sum_all_simulation_lines_weights = 0
    for i in range(2, len(gbcountRANDOM)):
        sum_all_simulation_lines_weights += i * i * gbcountRANDOM[i]

    # measured with weights - in the long term
    sum_all_measured_lines_weights = 0
    for value, frequency in value_counts.items():
        sum_all_measured_lines_weights += value * value * frequency
    LongtermES = sum_all_measured_lines_weights / sum_all_simulation_lines_weights
    print("Weights Ratio in the long term\n Measured (E) \ Simulation (S): ", LongtermES)

    return PercentagePS, PercentagePM, PercentagePMS, LongtermES, sum_all_measured_lines, sum_all_simulation_lines


# Function to calculate the width and height of the image based on monitor resolution
def calculate_image_size(image, padding=10):
    monitor_width = root.winfo_screenwidth()
    monitor_height = root.winfo_screenheight()
    max_width = (monitor_width - (NUMBER_OF_PHOTOS + 1) * padding) // NUMBER_OF_PHOTOS
    max_height = monitor_height - 2 * padding

    width, height = image.size

    # Calculate the new dimensions while maintaining the aspect ratio
    if width > max_width:
        height = int(max_width * height / width)
        width = max_width

    if height > max_height:
        width = int(max_height * width / height)
        height = max_height

    image = image.resize((width, height), Resampling.LANCZOS)
    # Convert the image to a PhotoImage object
    photo = ImageTk.PhotoImage(image)
    return photo


######################################################################################################################################################################################

# Create the main window
root = tk.Tk()
root.title("SOA 2.0")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to fit the screen
window_width = int(screen_width * 0.7)
window_height = int(screen_height * 0.5)
root.geometry(f"{window_width}x{window_height}")

# Create a frame for buttons and sliders at the bottom
bottom_frame = tk.Frame(root)
bottom_frame.pack(side="bottom", fill="x")

# Create buttons
load_button = tk.Button(bottom_frame, text="(1) Load Image", font=("Arial", 16), command=load_image, bg="blue",
                        fg="white")
load_button.pack(side=tk.LEFT)

# Create sliders
slider1_var = tk.DoubleVar()
slider1_label = tk.Label(bottom_frame, text="<=THRESH", font=("Arial", 16))
slider1 = tk.Scale(bottom_frame, variable=slider1_var, from_=0, to=10, orient="horizontal",
                   command=lambda _: update_slider_value1(slider1_var, slider1_label))
slider1.pack(side=tk.LEFT)
slider1_label.pack(side=tk.LEFT)

# Branch Size
slider2_label = tk.Label(bottom_frame, text="<=Branch Size", font=("Arial", 16))
slider2_var = tk.DoubleVar()
slider2 = tk.Scale(bottom_frame, variable=slider2_var, from_=0, to=10, orient="horizontal",
                   command=lambda _: update_slider_value2(slider2_var, slider2_label))
slider2.pack(side=tk.LEFT)
slider2_label.pack(side=tk.LEFT)

"""slider3_label = tk.Label(bottom_frame, text="<=Angle", font=("Arial", 16))
slider3_var = tk.DoubleVar()
slider3 = tk.Scale(bottom_frame, variable=slider3_var, from_=0, to=10, orient="horizontal",
                   command=lambda _: update_slider_value3(slider3_var, slider3_label))
slider3.pack(side=tk.LEFT)
slider3_label.pack(side=tk.LEFT)"""

# Create buttons
segmentation_button = tk.Button(bottom_frame, text="(2) Create Segmentation", font=("Arial", 16),
                                command=create_segmentation2, bg="blue", fg="white")
segmentation_button.pack(side=tk.LEFT)

# Create buttons
results_button = tk.Button(bottom_frame, text="(3) Plot Dendrites", font=("Arial", 16), command=generate_results,
                           bg="blue", fg="white")
results_button.pack(side=tk.LEFT)

# Create buttons
export_button = tk.Button(bottom_frame, text="(4) Save Statistics", font=("Arial", 16), command=write_results, bg="red",
                          fg="white")
export_button.pack(side=tk.LEFT)

# Create a frame for images
image_frame = tk.Frame(root)
image_frame.pack(side="top", fill="both", expand=True)

# Create two frames for images
left_frame = tk.Frame(image_frame, width=200, height=400)
left_frame.pack(side=tk.LEFT, fill="both", expand=True)
# Create a canvas to display the input image in the left frame
left_frame.image_label = tk.Label(left_frame)
left_frame.image_label.pack(fill="both", expand=True)

second_input_frame = tk.Frame(image_frame, width=400, height=400)
second_input_frame.image_label = tk.Label(second_input_frame)
second_input_frame.image_label.pack(fill="both", expand=True)
second_input_frame.pack(side=tk.LEFT, fill="both", expand=True)

right_frame = tk.Frame(image_frame, width=400, height=400)
right_frame.pack(side=tk.LEFT, fill="both", expand=True)
# Create a canvas to display the segmented image in the right frame
right_frame.image_label = tk.Label(right_frame)
right_frame.image_label.pack(fill="both", expand=True)

Ex_right_frame = tk.Frame(image_frame, width=400, height=400)
Ex_right_frame.pack(side=tk.LEFT, fill="both", expand=True)
# Create a canvas to display the segmented image in the right frame
Ex_right_frame.image_label = tk.Label(Ex_right_frame)
Ex_right_frame.image_label.pack(fill="both", expand=True)

# Start the Tkinter main loop
root.mainloop()
######################################################################################################################
