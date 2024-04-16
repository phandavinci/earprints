import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

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
    
    class_mapping = {i: str(i) for i in range(num_classes)}
    train_df['labels'] = train_df['labels'].map(class_mapping)
    test_df['labels'] = test_df['labels'].map(class_mapping)

    # Preprocess input images
    train_datagen = ImageDataGenerator(rescale=1./255)

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

    return train_generator, test_generator

def train_model(dataset_dir, img_height, img_width, batch_size, num_classes, epochs):
    # Load data and preprocess
    train_generator, test_generator = load_and_preprocess_data(dataset_dir, img_height, img_width, batch_size)

    # Create base model
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',  # Load pre-trained weights
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)  # Add BatchNormalization layer
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine base model and custom head
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    def training_progress(epoch, logs, test_generator):
        tqdm.write(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Accuracy: {logs["accuracy"]:.4f} - Validation Loss: {logs["val_loss"]:.4f} - Validation Accuracy: {logs["val_accuracy"]:.4f}')
        
        # Calculate and display metrics
        y_true = []
        y_pred = []
        for _ in range(len(test_generator)):
            batch_x, batch_y = next(test_generator)
            y_true.extend(np.argmax(batch_y, axis=1))
            y_pred.extend(np.argmax(model.predict(batch_x), axis=1))
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        tqdm.write(f'Epoch {epoch + 1}/{epochs} - Precision: {precision:.4f} - Recall: {recall:.4f}')

    callbacks = [tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: training_progress(epoch, logs, test_generator)
    ), EarlyStopping(monitor='val_loss', patience=3)]
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks = callbacks
    )

    # Save the trained model
    model.save('earprintsWeights.h5')

    # Save training history
    np.save('training_history.npy', history.history)

    print("Model training completed.")
    
dataset_dir = '../dataset'
img_height, img_width = 224, 224
batch_size = 32
num_classes = 4
epochs = 15

train_model(dataset_dir, img_height, img_width, batch_size, num_classes, epochs)
