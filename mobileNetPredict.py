import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('ear_classification_model.h5')

# Function to preprocess input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))  # Resize the image to match the input size
    img = img / 255.0  # Normalize the image
    return img

# Function to predict the person
def predict_person(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    person_index = np.argmax(prediction)
    return person_index

# Example usage
image_path = 'test_ear_image.jpg'  # Path to the ear image
person_index = predict_person(image_path)
print("Predicted person:", os.listdir('data')[person_index])
