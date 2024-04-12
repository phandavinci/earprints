import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set your dataset directory
dataset_dir = '../dataset'

# Generate lists of image paths and labels
image_paths = []
labels = []

for class_label, class_name in enumerate(sorted(os.listdir(dataset_dir))):
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image_paths.append(image_path)
        labels.append(class_label)

# Split the data into training and testing sets
train_image_paths, _, train_labels, _ = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Define image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.7, 1.3],  # Randomly change brightness
    zoom_range=[0.9, 1.1],  # Randomly zoom in and out
    width_shift_range=0.1,  # Shift width
    height_shift_range=0.1,  # Shift height
    rotation_range=20,  # Rotate images up to 20 degrees
    fill_mode='nearest',  # Fill mode for points outside the input boundaries
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,  # Preprocess input according to MobileNetV2 requirements
)

# Define progress bars for each class
progress_bars = [tqdm(total=len(train_image_paths)//4, desc=f"Class {i}") for i in range(4)]

# Apply augmentation and save images
for image_path, label in zip(train_image_paths, train_labels):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)

    # Generate augmented images
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=os.path.dirname(image_path), save_prefix='aug', save_format='jpg')):
        if i >= 4:  # Generate 5 augmented images per original image
            break
        # Update progress bar for the corresponding class
        progress_bars[label].update(1)

# Close progress bars
for progress_bar in progress_bars:
    progress_bar.close()

print("Image augmentation completed.")
