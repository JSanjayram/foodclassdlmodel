import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from model import create_cnn_model
from config import *

def load_data():
    """Load and preprocess food images"""
    data, labels = [], []
    
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} not found. Creating sample data...")
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append(img)
                labels.append(idx)
    
    return np.array(data, dtype="float32") / 255.0, np.array(labels)

def train_model():
    """Train the food classification model"""
    print("Loading dataset...")
    
    # Check if data exists, if not create sample data
    if not any(os.path.exists(os.path.join(TRAIN_DIR, cls)) for cls in CLASS_NAMES):
        print("No training data found. Please add images to data/train/ folders.")
        return
    
    data, labels = load_data()
    
    if len(data) == 0:
        print("No images found. Please add training images.")
        return
    
    print(f"Loaded {len(data)} images")
    
    # Convert labels to categorical
    labels = to_categorical(labels, NUM_CLASSES)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Create model
    print("Creating model...")
    model = create_cnn_model()
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    train_model()