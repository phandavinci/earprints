import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Define custom triplet loss
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, alpha):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def call(self, y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.maximum(positive_distance - negative_distance + self.alpha, 0.0)

def train_model(dataset_dir, img_height, img_width, batch_size, num_classes, epochs, alpha):
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
    
    class_mapping = {i: str(i) for i in range(num_classes)}
    train_df['labels'] = train_df['labels'].map(class_mapping)
    test_df['labels'] = test_df['labels'].map(class_mapping)

    # Preprocess input images
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=7,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.85, 1.25],  # Randomly change brightness
        zoom_range=[0.95, 1.1], 
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_paths",
        y_col="labels",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_paths",
        y_col="labels",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )


    # Create base model
    base_model = MobileNetV2(
        weights=None,  # Remove pre-trained weights
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine base model and custom head
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model with triplet loss
    model.compile(optimizer=Adam(), loss=TripletLoss(alpha=alpha))

    # Define callbacks for training progress and evaluation metrics
    class TrainingProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            tqdm.write(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Validation Loss: {logs["val_loss"]:.4f}')

    class Metrics(tf.keras.callbacks.Callback):
        def __init__(self, test_generator):
            super(Metrics, self).__init__()
            self.test_generator = test_generator

        def on_epoch_end(self, epoch, logs=None):
            y_true = []
            y_pred = []
            for _ in range(len(self.test_generator)):
                batch_x, batch_y = next(self.test_generator)
                y_true.extend(np.argmax(batch_y, axis=1))
                y_pred.extend(np.argmax(model.predict(batch_x), axis=1))
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            tqdm.write(f'Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[TrainingProgress(), Metrics(test_generator)]
    )

    # Save the trained model
    model.save('mobilenetv2_custom.h5')

    # Save training history
    np.save('training_history.npy', history.history)

    print("Model training completed.")

# Example usage
dataset_dir = '../dataset'
img_height, img_width = 224, 224
batch_size = 32
num_classes = 4
epochs = 10
alpha = 0.2  # Margin for triplet loss

train_model(dataset_dir, img_height, img_width, batch_size, num_classes, epochs, alpha)
