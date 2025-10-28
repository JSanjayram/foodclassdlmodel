#!/usr/bin/env python3
"""
Food Image Classification - Main Runner
"""
import subprocess
import sys
import os

def check_model():
    """Check if model exists"""
    if not os.path.exists('food_classifier.h5'):
        print("❌ Model not found!")
        print("Run 'python train.py' to train the model first.")
        return False
    return True

def run_app():
    """Run the Streamlit app"""
    print("🍕 Starting Food Classification App...")
    print("📱 Access at: http://localhost:8501")
    
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port=8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🍕 Food Image Classification System")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. 📊 Train Model")
        print("2. 🚀 Run Web App")
        print("3. 🔍 Test Prediction")
        print("4. 📁 Download Sample Data")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("Training model...")
            subprocess.run([sys.executable, "train.py"])
        
        elif choice == "2":
            if check_model():
                run_app()
        
        elif choice == "3":
            if check_model():
                img_path = input("Enter image path: ").strip()
                if os.path.exists(img_path):
                    subprocess.run([sys.executable, "predict.py", img_path])
                else:
                    print("Image not found!")
        
        elif choice == "4":
            subprocess.run([sys.executable, "data_downloader.py"])
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()