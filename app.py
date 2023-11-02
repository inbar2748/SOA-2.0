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

######################################################################################################################################################################################
######################################################################################################################################################################################
BRANCHLENGTH_THRESHOLD = 5
NUMBER_OF_PHOTOS = 4
global contours3
global BRLengths
global Logs_df
global BL_df
global Angles_df
global actual_image
global segment_image
global skele_image
BRLengths = []
contours3 = []
Logs = []
CC=[]


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
    Angles.clear()
    Logs.clear()
    BINOMIAL_LINES.clear()

    CC = create_segmentation()
    # Call the function to find unique rows in contours2
    # unique_contours = find_unique_rows_in_contours(CC)
    print("Retained branches after some/non-dendrites using Branch Length THRESHOLD================" + str(len(CC)))


######################################################################################################################################################################################
def save_Framed_images():
    folder_path, file_name_with_extension = os.path.split(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name_with_extension)
    new_file_savenameOI = file_name_without_extension + "_OriginalImage.png"
    actual_image.save(new_file_savenameOI)
    new_file_savenameSI = file_name_without_extension + "_BW_Segmentation.png"
    segment_image.save(new_file_savenameSI)
    new_file_savenameBWS = file_name_without_extension + "_BW_Skeleton.png"
    skele_image.save(new_file_savenameBWS)
    print(" ALL Images saved in Base Directory ")


######################################################################################################################################################################################    
# Define a custom function to find the nearest point
def find_nearest_point(point, curve1):
    distances = np.sqrt(np.sum((curve1 - point) ** 2, axis=1))
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    return curve1[min_index], min_index, min_distance


######################################################################################################################################################################################
global Angles, DD, BINOMIAL_LINES
Angles = []
DD = []
Logs = []
BINOMIAL_LINES = []


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

    fig = plt.figure();
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
        plt.title(" Actual Image with overlaid detected Branches")
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
        print(f"Length of Branch{ii} is   {BranchLen}")
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
    # it can adapt this code according to your actual data structure.

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

    # Define the size of the plane
    plane_size = (512, 512)
    # Number of lines to generate
    num_lines = len(DD)
    # Generate random lines using binomial distribution
    for _ in range(num_lines):
        # Generate random endpoints for the line
        x1, y1 = np.random.randint(0, plane_size[0]), np.random.randint(0, plane_size[1])
        x2, y2 = np.random.randint(0, plane_size[0]), np.random.randint(0, plane_size[1])

        # Create a 2D array representing the line
        line = np.array([[x1, y1], [x2, y2]])

        # Append the line to the list
        BINOMIAL_LINES.append(line)

    # Print the generated lines
    # for i, line in enumerate(EE):
    #    print(f"Line {i + 1}:")
    #    print(line)

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

    def group_parallel_branches(contours, angle_threshold=10):
        global angles
        global gbcount

        angles = []
        gbcount = []

        for contour in contours:
            first_point = contour[0]
            last_point = contour[-1]
            slope = compute_slope(first_point, last_point)
            angle = np.arctan(slope) * 180 / np.pi
            angles.append(angle)

        grouped_branches = {}
        visited = [False] * len(contours)

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

                cumulative_lengths[i + 1] = cumulative_length

            if not visited[i]:
                group = [i + 1]
                visited[i] = True
                for j, angle2 in enumerate(angles):
                    if i != j and not visited[j] and abs(angle1 - angle2) <= angle_threshold:
                        group.append(j + 1)
                        visited[j] = True
                grouped_branches[i + 1] = group

        return grouped_branches, cumulative_lengths

    # Calculate parallel groups and get cumulative lengths
    parallel_groups, cumulative_lengths = group_parallel_branches(contours, angle_threshold=angle_threshold)

    # Count the number of branches in each parallel group
    group_branch_counts = {group: len(branches) for group, branches in parallel_groups.items()}

    # Count the total number of parallel groups
    num_parallel_groups = len(parallel_groups)

    # Print parallel group information, including angles, branch coordinates, and branch counts
    for group, branches in parallel_groups.items():
        print(f"-------------------------Parallel Group information ---------------------------------:")
        print(f"Number of Parallel Lines in this Group --------: {group_branch_counts[group]}")
        thisgcount = group_branch_counts[group]
        gbcount.append(thisgcount)
        for branch in branches:
            print(f"  Key {branch}:")
            branch_coordinates = contours[branch - 1]
            first_point = branch_coordinates[0]
            last_point = branch_coordinates[-1]
            slope = compute_slope(first_point, last_point)
            angle = np.arctan(slope) * 180 / np.pi
            print(f"    Angle: {angle:.2f} degrees" + f" First Point: {first_point}" + f" Last Point: {last_point}")

            # Print the cumulative length for the branch
            if branch in cumulative_lengths:
                length = cumulative_lengths[branch]
                print(f"    Cumulative Length: {length:.2f} units")

    # Optionally, visualize the branches as well
    for i, contour in enumerate(contours):
        x, y = contour[:, 0], contour[:, 1]
        plt.plot(x, y, label=f"Branch {i + 1}")

    # option angles check
    for i, angle in enumerate(angles):
        print(f"Angle {i + 1}: {angle} degrees")

    return angles, num_parallel_groups, gbcount, contours
    # Print the values of angles to the console


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
    slider_value3 = slider3_var.get()  # Replace with the actual function to get the slider value
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
    }

    # Check if the slider value is in the mapping
    if slider_value3 in threshold_mapping3:
        ANG_TH = threshold_mapping3[slider_value3]
    else:
        # Default threshold value for other cases
        ANG_TH = 10  # can set a default value here if needed

    # Angles, num_groups, gbcount,Mcontours=analyze_and_plot_parallel_branches(DD, angle_threshold=ANG_TH)
    Angles, num_groups, gbcount, Mcontours = analyze_and_plot_parallel_branches(DD, angle_threshold=ANG_TH)
    # Modify negative values in Angles
    Angles = [abs(angle) for angle in Angles]
    print(Angles)
    for i in range(len(Angles)):
        if Angles[i] < 0:
            Angles[i] = 360 + Angles[i]

    fig = plt.figure();
    plt.imshow(actual_image)
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    plt.plot([x1, x2], [y1, y2])
    # Optionally, visualize the branches as well
    for i, contour in enumerate(Mcontours):
        x, y = contour[:, 0], contour[:, 1]
        plt.plot(x, y, '-y', linewidth=3)
        plt.title(" Measured Branch Estimation Plotted over 2D image")
    #       plt.plot(XY[:, 0], f, '-y', linewidth=3)

    plt.show()

    # Count the frequency of each unique value
    value_counts = {}
    for value in gbcount:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    # Extract unique values and their frequencies and sort them
    unique_values = sorted(list(value_counts.keys()))
    frequencies = [value_counts[value] for value in unique_values]
    plt.figure()
    # Create a bar graph
    plt.bar(range(len(unique_values)), frequencies)
    plt.xticks(range(len(unique_values)), unique_values)

    # Add labels and title
    plt.xlabel('Parallel Groups with Branches')
    plt.ylabel('Frequency Count for respective groups')
    plt.title("DETECTED-Total branches== " + str(sum(gbcount)) + ", Arranged into" + str(len(gbcount)) + " groups")
    plt.show()

    print(
        "--------------------------- AFTER MEASURED BRANCH ANALYSIS, NOW BELOW IS THE INFORMATION FOR SIMULATED RANDOM LINES----------------------------------------")

    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    # Parameters for the binomial distribution
    num_lines = sum(gbcount)
    n = len(np.unique(gbcount))  # Number of trials
    p = (1 / n)  # Probability of success in each trial

    def create_binomial_lines(n, p, num_lines):

        # Dimensions of the 2D plane
        dimensions = 512

        # Initialize an empty list to store line coordinates
        lines = []

        # Simulate drawing lines based on the binomial distribution
        for _ in range(num_lines):
            # Determine the number of successful trials (lines) using the binomial distribution
            k = np.random.binomial(n, p)

            # Generate random start and end points for a line within the specified dimensions
            for _ in range(k):
                start_point = np.random.randint(0, dimensions, 2)
                end_point = np.random.randint(0, dimensions, 2)

                # Create a NumPy array for the line coordinates
                line_array = np.array([start_point, end_point])

                # Add the line array to the list
                lines.append(line_array)

        return lines

    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    ######################################################################################################################################################################################
    #    BINOMIAL_LINES=create_binomial_lines(n,p,num_lines)
    #########################################################################################################

    AnglesRANDOM, num_groupsRANDOM, gbcountRANDOM, contoursRANDOM = analyze_and_plot_parallel_branches(BINOMIAL_LINES,
                                                                                                       angle_threshold=ANG_TH)

    fig = plt.figure();
    plt.imshow(actual_image)
    x1, y1 = line_coordinates[0]
    x2, y2 = line_coordinates[1]
    plt.plot([x1, x2], [y1, y2])
    # Optionally, visualize the branches as well
    for i, contour in enumerate(contoursRANDOM):
        x, y = contour[:, 0], contour[:, 1]
        plt.plot(x, y, '-y', linewidth=3)
        plt.title(" Random Branch Estimation Plotted over 2D image")

    plt.show()

    value_countsR = {}
    for valueR in gbcountRANDOM:
        if valueR in value_countsR:
            value_countsR[valueR] += 1
        else:
            value_countsR[valueR] = 1

    # Extract unique values and their frequencies and sort them
    unique_valuesR = sorted(list(value_countsR.keys()))
    frequenciesR = [value_countsR[valueR] for valueR in unique_valuesR]
    plt.figure()
    # Create a bar graph
    plt.bar(range(len(unique_valuesR)), frequenciesR)
    plt.xticks(range(len(unique_valuesR)), unique_valuesR)

    # Add labels and title
    plt.xlabel('Parallel Groups with Branches')
    plt.ylabel('Frequency Count for respective groups')
    plt.title(
        "RANDOM-Total branches== " + str(sum(gbcountRANDOM)) + ", Arranged into" + str(len(gbcountRANDOM)) + " groups")
    plt.show()

    # Count the frequency of each unique value
    value_counts = {}
    for value in gbcount:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

        # Print the value-frequency pairs

    print("STATISTICS for Measured Classification")
    for value, frequency in value_counts.items():
        print(f"{value}:{frequency}")

    # Count the frequency of each unique value
    value_countsR = {}
    for valueR in gbcountRANDOM:
        if valueR in value_countsR:
            value_countsR[valueR] += 1
        else:
            value_countsR[valueR] = 1

    # Print the value-frequency pairs
    print("STATISTICS for Random Classification")
    for valueR, frequency in value_countsR.items():
        print(f"{valueR}:{frequency}")

    disp_final_stats(asli_contours, BINOMIAL_LINES, gbcount, gbcountRANDOM)
    ######################################################################################################################################################################################

    # Extract the folder path from the file path
    folder_path, file_name_with_extension = os.path.split(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name_with_extension)
    new_file_name = file_name_without_extension + ".csv"
    csv_file_path = os.path.join(folder_path, new_file_name)

    Logs_df = pd.DataFrame(Logs)
    BL_df = pd.DataFrame(BRLengths)
    ANGLE_df = pd.DataFrame(Angles)
    BP_df = pd.DataFrame(branch_points)
    BranchesCompound_df = pd.DataFrame({num_objectsCompound})
    BranchesDetached_df = pd.DataFrame({num_objects2})
    BranchesRetained_df = pd.DataFrame({len(DD)})
    FileName_df = pd.DataFrame({file_path})

    result_df = pd.concat(
        [Logs_df, BL_df, ANGLE_df, BP_df, BranchesCompound_df, BranchesDetached_df, BranchesRetained_df, FileName_df],
        axis=1, ignore_index=True)

    # Create a sample DataFrame
    result_df2 = pd.DataFrame({'Current Branch': result_df[0][:],
                               'Relative Branch': result_df[1][:],
                               'ClosestX': result_df[2][:],
                               'ClosestY': result_df[3][:],
                               'Shortest Distance': result_df[4][:],
                               'Branch Length': result_df[5][:],
                               'Branch Angle Base': result_df[6][:],
                               'Branch PointsX': result_df[7][:],
                               'Branch PointsY': result_df[8][:],
                               'Compound Branches': result_df[9][:],
                               'Detached Branches': result_df[10][:],
                               'Cleaned Retained Branches': result_df[11][:],
                               'Image File Processed': result_df[12][:]})

    # Save the DataFrame to a CSV file
    result_df2.to_csv(csv_file_path, index=False)
    print(f'Data has been saved to {csv_file_path}')
    save_Framed_images()

    create_graphs(BRLengths, Angles)


######################################################################################################################################################################################

def create_graphs(BRLengths, Angles):
    plt.figure()
    # Create a bar graph for branch lengths
    plt.bar(range(len(BRLengths)), BRLengths)
    # Add labels and a title
    plt.xlabel(' Branches')
    plt.ylabel('Length in Unit Pixels')
    plt.title('Branch Length Distribution')
    # Calculate and display average and standard deviation for branch lengths
    avg_length = np.mean(BRLengths)
    std_dev_length = np.std(BRLengths)
    plt.text(len(BRLengths), max(BRLengths) , f'Avg: {avg_length:.2f}\nStd Dev: {std_dev_length:.2f}',color='black',
             fontsize=12, bbox=dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5), horizontalalignment='right')
    # Show the graph
    plt.show()

    # Create a bar graph for branch angles
    plt.figure()
    plt.bar(range(len(Angles)), Angles)
    # Add labels and a title
    plt.xlabel(' Branches')
    plt.ylabel('Angle in Degrees with Horizontal Base')
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
        plt.plot(x, y, label=f'Branch {i + 1}')
        plt.xlim(0, 512)
        plt.ylim(0, 512)
    # Add a legend to the plot
    plt.legend()
    ax.invert_yaxis()
    plt.title(" 2D Branches based on Image Segmentation")
    plt.show()


######################################################################################################################################################################################
def disp_final_stats(contours, BINOMIAL_LINES, gbcount, gbcountRANDOM):
    all_lines_detected_in_original_image = len(contours)
    sum_all_measured_lines = len(DD)
    sum_all_simulation_lines = len(BINOMIAL_LINES)

    PercentagePM = (sum_all_measured_lines / all_lines_detected_in_original_image) * 100
    PercentagePS = (sum_all_simulation_lines / all_lines_detected_in_original_image) * 100
    PercentagePMS = sum_all_measured_lines / sum_all_simulation_lines

    # Use Counter to count the occurrences of each value
    value_counts = Counter(gbcount)
    value_counts_dict = dict(value_counts)
    weights = []
    # Print the unique values and their occurrence frequency
    for value, frequency in value_counts_dict.items():
        thisweight = value * frequency
        weights.append(thisweight)
    NE = sum(weights)

    weightsR = []
    # Use Counter to count the occurrences of each value
    value_countsR = Counter(gbcountRANDOM)
    # Convert the result to a dictionary for easy access
    value_counts_dictR = dict(value_countsR)
    weightsR = []
    # Print the unique values and their occurrence frequency
    for valueR, frequencyR in value_counts_dictR.items():
        thisweightR = valueR * frequencyR
        weightsR.append(thisweightR)
    NS = sum(weightsR)

    NSERatio = NE / NS

    print(f" PM Percentage is ----------- {PercentagePM}")
    print(f" PS Percentage is ............{PercentagePS}")
    print(f" Weights Ratio computed is .... {NSERatio}")

    return


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
window_width = int(screen_width * 0.7)  # You can adjust the percentage as needed
window_height = int(screen_height * 0.5)  # You can adjust the percentage as needed
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

slider3_label = tk.Label(bottom_frame, text="<=Angle", font=("Arial", 16))
slider3_var = tk.DoubleVar()
slider3 = tk.Scale(bottom_frame, variable=slider3_var, from_=0, to=10, orient="horizontal",
                   command=lambda _: update_slider_value3(slider3_var, slider3_label))
slider3.pack(side=tk.LEFT)
slider3_label.pack(side=tk.LEFT)

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
