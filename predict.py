import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import CLASS_NAMES, MODEL_PATH, IMG_WIDTH, IMG_HEIGHT

def predict_food(image_path, model_path=MODEL_PATH):
    """Predict food type from image"""
    try:
        # Load model
        model = load_model(model_path)
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None, 0
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return CLASS_NAMES[class_idx], confidence
        
    except Exception as e:
        print(f"Error: {e}")
        return None, 0

def predict_from_array(img_array, model_path=MODEL_PATH):
    """Predict food type from numpy array"""
    try:
        model = load_model(model_path)
        
        # Preprocess
        img = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return CLASS_NAMES[class_idx], confidence
        
    except Exception as e:
        print(f"Error: {e}")
        return None, 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        food_type, confidence = predict_food(image_path)
        if food_type:
            print(f"Predicted: {food_type.title()} (Confidence: {confidence:.1%})")
        else:
            print("Could not predict food type")
    else:
        print("Usage: python predict.py <image_path>")