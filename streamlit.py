import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import shutil
from PIL import Image

# Load YOLO model
yolo_model = YOLO('yolo.pt')

# Load mobile model
model = load_model('earprintsWeights.h5', compile=False)

def load_and_process_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def mobile(image_path):
    img_array = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(img_array)
    print(predictions)
    class_index = np.argmax(predictions)
    print(class_index)
    # Get class names
    class_names = sorted(os.listdir('../dataset'))
    print(class_names)

    return class_names[class_index].capitalize()

def predict_class(image_path):
    images = []; predictions = []
    if os.path.exists('ear'): shutil.rmtree('ear')
    results = yolo_model(image_path)
    for i, result in enumerate(results):
        result.save_crop('')
        try:
            images.append(os.path.join(image_path,str(i)+'.jpg'))
            predictions.append(mobile('ear/im.jpg'+(str(i+1) if i>0 else '')+'.jpg'))
        except:
            continue
    return images, predictions


def main():
    st.title("Earprints Prediction")
    st.write("Upload one or more images to predict the class of each earprint.")
    if os.path.exists('uploaded'): shutil.rmtree('uploaded')
    os.makedirs('uploaded')
    uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            
            with open(f'uploaded/{i}.jpg', 'wb') as f:
                f.write(uploaded_file.getbuffer())
    images, predictions = predict_class("uploaded")
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        st.subheader(f"Image {i + 1}: Predicted Class - {prediction}")
        st.image(image, caption=f"{prediction}")

if __name__ == "__main__":
    main()
