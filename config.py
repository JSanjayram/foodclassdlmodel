# Configuration for Food Image Classification

# Model Configuration
INPUT_SIZE = (224, 224, 3)  # MobileNetV2 input size
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Data Configuration
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
MODEL_PATH = 'food_classifier_pretrained.h5'

# Class Labels
CLASS_NAMES = ['burger', 'fries', 'pasta', 'pizza']

# Image Processing
IMG_HEIGHT = 224
IMG_WIDTH = 224
VALIDATION_SPLIT = 0.2