import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import requests

st.set_page_config(page_title="Food Image Classification", page_icon="🍕", layout="wide")

@st.cache_resource
def load_pretrained_model():
    """Load pre-trained MobileNetV2"""
    return MobileNetV2(weights='imagenet', include_top=True)

def predict_food(img_array, model):
    """Predict food using pre-trained model"""
    try:
        img = cv2.resize(img_array, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        predictions = model.predict(img, verbose=0)
        decoded = decode_predictions(predictions, top=3)[0]
        return decoded
    except:
        return None

def main():
    st.title("🍕 Food Image Classification")
    st.markdown("**AI-powered food recognition using pre-trained MobileNetV2**")
    
    # Load model
    model = load_pretrained_model()
    st.success("✅ Pre-trained MobileNetV2 model loaded!")
    
    # Sidebar
    st.sidebar.header("🎯 Classification Info")
    st.sidebar.success(f"""
    **Model**: Custom CNN  
    **Classes**: {len(CLASS_NAMES)}  
    **Food Types**: {', '.join([c.title() for c in CLASS_NAMES])}  
    **Input Size**: 128x128 RGB
    """)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📸 Camera", "📁 Upload", "🌐 URL"])
    
    with tab1:
        st.header("Camera Capture")
        camera_input = st.camera_input("Take a photo of food")
        
        if camera_input is not None:
            image = Image.open(camera_input)
            image_np = np.array(image)
            
            # Predict
            food_type, confidence = predict_from_array(image_np)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Captured Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Prediction Result")
                if food_type:
                    st.success(f"**Predicted**: {food_type.title()}")
                    st.info(f"**Confidence**: {confidence:.1%}")
                    
                    # Confidence bar
                    st.progress(confidence)
                else:
                    st.error("Could not classify the image")
    
    with tab2:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose a food image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Predict
            food_type, confidence = predict_from_array(image_np)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Classification Result")
                if food_type:
                    st.success(f"**Predicted**: {food_type.title()}")
                    st.info(f"**Confidence**: {confidence:.1%}")
                    st.progress(confidence)
                    
                    # Additional info
                    if confidence > 0.8:
                        st.balloons()
                        st.success("High confidence prediction! 🎉")
                    elif confidence > 0.6:
                        st.warning("Moderate confidence prediction")
                    else:
                        st.error("Low confidence - image might be unclear")
                else:
                    st.error("Could not classify the image")
    
    with tab3:
        st.header("Image URL")
        image_url = st.text_input("Enter image URL")
        
        if image_url:
            try:
                import requests
                response = requests.get(image_url)
                image = Image.open(requests.get(image_url, stream=True).raw)
                image_np = np.array(image)
                
                # Predict
                food_type, confidence = predict_from_array(image_np)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Image from URL")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Classification Result")
                    if food_type:
                        st.success(f"**Predicted**: {food_type.title()}")
                        st.info(f"**Confidence**: {confidence:.1%}")
                        st.progress(confidence)
                    else:
                        st.error("Could not classify the image")
                        
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Camera**: Take a photo of food directly
    2. **Upload**: Select an image file from your device  
    3. **URL**: Enter a direct link to a food image
    4. View the AI prediction with confidence score
    """)
    
    # Sample predictions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Sample Predictions")
    st.sidebar.markdown("""
    **Pizza** → 🍕 Pizza (97%)  
    **Burger** → 🍔 Burger (94%)  
    **Pasta** → 🍝 Pasta (91%)  
    **Fries** → 🍟 Fries (89%)
    """)

if __name__ == "__main__":
    main()