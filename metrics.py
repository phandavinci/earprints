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
            labels.append(class_label)

    # Split the data into training and testing sets
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    train_df = pd.DataFrame({'image_paths': train_image_paths, 'labels': train_labels})
    test_df = pd.DataFrame({'image_paths': test_image_paths, 'labels': test_labels})
    
    class_mapping = {i: str(i) for i in range(4)}
    train_df['labels'] = train_df['labels'].map(class_mapping)
    test_df['labels'] = test_df['labels'].map(class_mapping)


    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',  # adjust this if you have more than two classes
        shuffle=False  # to ensure the predictions match the order of the files
    )

    return test_generator

def train_model(dataset_dir, img_height, img_width, batch_size):
    # Load data and preprocess
    test_generator = load_and_preprocess_data(dataset_dir, img_height, img_width, batch_size)
    # Load custom trained model
    model = load_model('earprintsWeights.h5')

    # Predict on test data
    y_true = test_generator.classes
    y_pred = model.predict_generator(test_generator)
    y_pred_binary = [1 if pred > 0.9 else 0 for pred in y_pred]

    # Calculate performance metrics
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred_binary)

    # Print performance measures
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Specificity: {specificity}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")

    # Print Classification Report
    report = classification_report(y_true, y_pred_binary)
    print(report)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Show the plots
    plt.show()
    
dataset_dir = '../dataset'
img_height, img_width = 224, 224
batch_size = 32

train_model(dataset_dir, img_height, img_width, batch_size)
