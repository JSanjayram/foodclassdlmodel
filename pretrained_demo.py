import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import requests

def load_pretrained_model():
    """Load pre-trained MobileNetV2 for demo"""
    model = MobileNetV2(weights='imagenet', include_top=True)
    return model

def predict_image(image_path, model):
    """Predict using pre-trained ImageNet model"""
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Predict
    predictions = model.predict(img)
    decoded = decode_predictions(predictions, top=3)[0]
    
    return decoded

def predict_from_url(url, model):
    """Predict from image URL"""
    try:
        response = requests.get(url)
        img = Image.open(requests.get(url, stream=True).raw)
        img = np.array(img)
        
        if len(img.shape) == 3:
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            predictions = model.predict(img)
            decoded = decode_predictions(predictions, top=3)[0]
            return decoded
    except:
        return None

def demo():
    """Demo with pre-trained model"""
    print("🍕 Food Classification Demo with Pre-trained MobileNetV2")
    print("=" * 50)
    
    # Load model
    print("Loading pre-trained MobileNetV2...")
    model = load_pretrained_model()
    print("✅ Model loaded!")
    
    # Sample food URLs for testing
    sample_urls = {
        "Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400",
        "Burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400",
        "Pasta": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=400"
    }
    
    print("\n🔍 Testing with sample images...")
    for food_name, url in sample_urls.items():
        print(f"\n--- {food_name} ---")
        predictions = predict_from_url(url, model)
        if predictions:
            for i, (imagenet_id, label, score) in enumerate(predictions):
                print(f"{i+1}. {label}: {score:.1%}")
        else:
            print("Could not process image")
    
    print("\n✨ Demo complete! The model can recognize many food items from ImageNet classes.")
    print("💡 For custom food classification, we would fine-tune this model with specific food data.")

if __name__ == "__main__":
    demo()