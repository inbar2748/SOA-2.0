import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL.Image import Resampling
from skimage.filters import frangi
from skimage.morphology import remove_small_objects
import os
from PIL import Image, ImageTk


######################################################################################################################################################################################
NUMBER_OF_PHOTOS = 2
global actual_image
global segment_image
global skele_image


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
    global input_image, file_path, skele_image, segment_image

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



######################################################################################################################################################################################
def save_framed_images():
    folder_path, file_name_with_extension = os.path.split(file_path)
    file_name_without_extension, file_extension = os.path.splitext(file_name_with_extension)

    # Save the segmented image
    new_file_savenameSI = file_name_without_extension + "_BW_Segmentation.png"
    segment_image.save(os.path.join(folder_path, new_file_savenameSI))

    print(" ALL Images saved in Base Directory ")



######################################################################################################################################################################################

def update_slider_value1(slider_var, label):
    slider_value = slider_var.get()
    # label.config(text=f"Slider Value: {slider_value}")
    # print(f"THRESHOLD: {slider_value}")

######################################################################################################################################################################################
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
slider1_label = tk.Label(bottom_frame, text="Threshold", font=("Arial", 16))
slider1 = tk.Scale(bottom_frame, variable=slider1_var, from_=0, to=10, orient="horizontal",
                   command=lambda _: update_slider_value1(slider1_var, slider1_label))
slider1.pack(side=tk.LEFT)
slider1_label.pack(side=tk.LEFT)

# Create buttons
segmentation_button = tk.Button(bottom_frame, text="(2) Create Segmentation", font=("Arial", 16),
                                command=create_segmentation, bg="blue", fg="white")
segmentation_button.pack(side=tk.LEFT)

# Save Image
save_segmentation_image = tk.Button(bottom_frame, text="(3) Save Image", font=("Arial", 16),
                                command=save_framed_images, bg="blue", fg="white")
save_segmentation_image.pack(side=tk.LEFT)

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

# Start the Tkinter main loop
root.mainloop()
######################################################################################################################
