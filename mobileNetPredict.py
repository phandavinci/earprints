import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import shutil
model = YOLO('yolo.pt')


        

def mobile(image_path):
    # Load the trained model
    model = load_model('earprintsWeights.h5', compile=False)
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make predictions
    predictions = model.predict(img_array)
    print(predictions)
    class_index = np.argmax(predictions)
    print(class_index)
    # Get class names
    class_names = sorted(os.listdir('../dataset'))
    print(class_names)

    print('Predicted Class:'+class_names[class_index])

def predict_class(image_path):
    if os.path.exists('ear'): shutil.rmtree('ear')
    results = model(image_path)
    for i, result in enumerate(results):
        result.save_crop('')
        mobile('ear/im.jpg'+(str(i+1) if i>0 else '')+'.jpg')

def detect_ear_and_class():
    # Function to detect ear and return the class
    pass  # To be implemented

# Example usage for predicting class of a single image
image_path = 'testing'
predict_class(image_path)

