import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os

# 1. Load the Map
print("📂 Loading training_data.csv...")
df = pd.read_csv('data/training_data.csv')

# 2. Preprocessing Function
def load_and_preprocess_image(image_name):
    img_path = os.path.join('data/UTKFace', image_name)
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) 
    return img / 255.0 

# 3. Build the CNN Brain
print("🧠 Building CNN Architecture...")
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)), # Fixed the 'input_shape' warning
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid', name='gender_output') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Data Loading (Testing with a subset first to ensure it works)
X = []
y = []

print("🔄 Loading first 1000 images for training...")
for img_name in df['image'][:1000]: 
    processed_img = load_and_preprocess_image(img_name)
    if processed_img is not None:
        X.append(processed_img)
        # Assuming gender is at index 1 based on your previous 'prepare_data' output
        # Update: We take the gender column from our CSV
        # Let's use the actual CSV data:
        pass 

# Get labels from CSV for the first 1000
y = df['gender'][:1000].values

X = np.array(X)
y = np.array(y)

# 5. THE ACTUAL TRAINING COMMAND (This makes the script 'Work')
print(f"🚀 TRAINING STARTING NOW on {len(X)} images...")
model.fit(X, y, epochs=10, batch_size=32)

# 6. Save the results
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/gender_model.h5')
print("✅ Success! Model saved in 'models/gender_model.h5'")