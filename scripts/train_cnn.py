import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
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

# 3. Build the Multi-Output CNN (Functional API)
print("🧠 Building Multi-Output CNN Architecture...")
inputs = Input(shape=(128, 128, 3))

# Shared Convolutional Base
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x) # Helps prevent overfitting

# Branch 1: Gender (Classification - Sigmoid)
gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)

# Branch 2: Age (Regression - ReLU/Linear)
age_output = layers.Dense(1, activation='linear', name='age_output')(x)

# Combine into one model
model = models.Model(inputs=inputs, outputs=[gender_output, age_output])

# Compile with two different loss functions
model.compile(
    optimizer='adam',
    loss={'gender_output': 'binary_crossentropy', 'age_output': 'mse'},
    metrics={'gender_output': 'accuracy', 'age_output': 'mae'}
)

# 4. Data Loading
X = []
y_gender = []
y_age = []

# Using 2000 images for better age variety (You can increase this to [:] for full training)
print("🔄 Loading 2000 images for multi-task training...")
for i, row in df.head(10000).iterrows(): 
    processed_img = load_and_preprocess_image(row['image'])
    if processed_img is not None:
        X.append(processed_img)
        y_gender.append(row['gender'])
        y_age.append(row['age'])

X = np.array(X)
y_gender = np.array(y_gender)
y_age = np.array(y_age)

# 5. THE TRAINING COMMAND
print(f"🚀 TRAINING STARTING NOW on {len(X)} images...")
model.fit(
    X, 
    {'gender_output': y_gender, 'age_output': y_age}, 
    epochs=30, 
    batch_size=32,
    validation_split=0.1
)

# 6. Save the results
if not os.path.exists('models'):
    os.makedirs('models')

# Save as a combined model
model.save('models/age_gender_model.h5')
print("✅ Success! Combined Model saved in 'models/age_gender_model.h5'")