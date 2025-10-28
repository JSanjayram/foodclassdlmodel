import os
import requests
from PIL import Image
import numpy as np
from config import TRAIN_DIR, CLASS_NAMES

def download_sample_images():
    """Download sample food images for testing"""
    
    # Sample URLs for each food type (placeholder - replace with actual URLs)
    sample_urls = {
        'pizza': [
            'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400',
            'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400'
        ],
        'burger': [
            'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400',
            'https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=400'
        ],
        'pasta': [
            'https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=400',
            'https://images.unsplash.com/photo-1621996346565-e3dbc353d2e5?w=400'
        ],
        'fries': [
            'https://images.unsplash.com/photo-1576107232684-1279f390859f?w=400',
            'https://images.unsplash.com/photo-1541592106381-b31e9677c0e5?w=400'
        ]
    }
    
    print("Downloading sample images...")
    
    for food_type, urls in sample_urls.items():
        food_dir = os.path.join(TRAIN_DIR, food_type)
        os.makedirs(food_dir, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    img_path = os.path.join(food_dir, f"{food_type}_{i+1}.jpg")
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {img_path}")
                else:
                    print(f"Failed to download: {url}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")

def create_synthetic_data():
    """Create synthetic food images for testing"""
    print("Creating synthetic training data...")
    
    colors = {
        'pizza': [(255, 200, 100), (200, 100, 50)],  # Orange/brown
        'burger': [(139, 69, 19), (255, 215, 0)],    # Brown/yellow
        'pasta': [(255, 255, 224), (255, 215, 0)],   # Light yellow
        'fries': [(255, 215, 0), (255, 165, 0)]      # Golden
    }
    
    for food_type in CLASS_NAMES:
        food_dir = os.path.join(TRAIN_DIR, food_type)
        os.makedirs(food_dir, exist_ok=True)
        
        # Create 20 synthetic images per class
        for i in range(20):
            # Create random colored image
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            
            # Add food-specific color patterns
            color1, color2 = colors[food_type]
            
            # Add some patterns
            for _ in range(5):
                x, y = np.random.randint(0, 128, 2)
                radius = np.random.randint(10, 30)
                cv2.circle(img, (x, y), radius, color1, -1)
            
            for _ in range(3):
                x, y = np.random.randint(0, 128, 2)
                radius = np.random.randint(5, 15)
                cv2.circle(img, (x, y), radius, color2, -1)
            
            # Save image
            img_path = os.path.join(food_dir, f"synthetic_{food_type}_{i+1}.jpg")
            cv2.imwrite(img_path, img)
        
        print(f"Created 20 synthetic images for {food_type}")

if __name__ == "__main__":
    import cv2
    
    choice = input("Choose option:\n1. Download sample images\n2. Create synthetic data\nEnter (1 or 2): ")
    
    if choice == "1":
        download_sample_images()
    elif choice == "2":
        create_synthetic_data()
    else:
        print("Invalid choice")
    
    print("Data preparation complete!")