import os
import cv2
from tqdm import tqdm

# Set your dataset directory
dataset_dir = '../dataset'

# Define target image size for MobileNetV2
target_size = (224, 224)

# Initialize a progress bar
num_images = sum(len(files) for _, _, files in os.walk(dataset_dir))
progress_bar = tqdm(total=num_images, desc="Resizing images")

# Iterate through each image in the dataset directory
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Read and resize the image
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, target_size)
        
        # Overwrite the original image with the resized image
        cv2.imwrite(image_path, resized_image)
        
        # Update the progress bar
        progress_bar.update(1)

# Close the progress bar
progress_bar.close()

print("Image resizing completed.")
