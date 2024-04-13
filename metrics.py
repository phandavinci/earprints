import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, matthews_corrcoef
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import os
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(dataset_dir, img_height, img_width, batch_size):
    # Generate lists of image paths and labels
    image_paths = []
    labels = []

    for class_label, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(str(class_label))

    # Split the data into training and testing sets
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.3, random_state=42)

    train_df = pd.DataFrame({'image_paths': train_image_paths, 'labels': train_labels})
    test_df = pd.DataFrame({'image_paths': test_image_paths, 'labels': test_labels})

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_paths",
        y_col="labels",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return test_generator

def train_model(dataset_dir, img_height, img_width, batch_size):
    # Load data and preprocess
    test_generator = load_and_preprocess_data(dataset_dir, img_height, img_width, batch_size)
    # Load custom trained model
    model = load_model('earprintsWeights.h5')
    test_images, test_labels = test_generator.next()
    y_true = np.argmax(test_labels, axis=1)  # Convert one-hot encoded labels to categorical labels
    y_pred = model.predict(test_images)
    y_pred_binary = [np.argmax(pred) for pred in y_pred]

    report = classification_report(y_true, y_pred_binary, output_dict=True)

    # Plot precision, recall, and F1-score for each class
    metrics = ['precision', 'recall', 'f1-score']
    classes = [str(i) for i in range(4)]  # Assuming class labels are integers
    for metric in metrics:
        values = [report[label][metric] for label in classes]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=classes, y=values, palette='viridis')
        plt.title(f'{metric.capitalize()} for each class')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
dataset_dir = '../dataset'
img_height, img_width = 224, 224
batch_size = 32

train_model(dataset_dir, img_height, img_width, batch_size)
