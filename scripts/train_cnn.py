import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
import cv2
import os

print("📂 Loading training_data.csv...")
df = pd.read_csv('data/training_data.csv')

# Shuffle the dataset to ensure mixed batches
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_and_preprocess_image(image_name):
    img_path = os.path.join('data/UTKFace', image_name)
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255.0 

print("🧠 Building Advanced Multi-Output CNN Architecture...")
inputs = Input(shape=(128, 128, 3))

# --- DATA AUGMENTATION ---
aug = layers.RandomFlip("horizontal")(inputs)
aug = layers.RandomRotation(0.1)(aug)        
aug = layers.RandomZoom(0.1)(aug)            
aug = layers.RandomContrast(0.1)(aug) # New: Help model handle different lighting

# --- DEEPER CNN ARCHITECTURE ---
# Block 1
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(aug)
x = layers.BatchNormalization()(x) # New: Stabilizes learning
x = layers.MaxPooling2D((2, 2))(x)

# Block 2
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 3
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# Block 4 (New)
x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)

# --- GENDER BRANCH ---
g = layers.Dense(128, activation='relu')(x)
g = layers.Dropout(0.5)(g)
gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(g)

# --- AGE BRANCH ---
a = layers.Dense(128, activation='relu')(x)
a = layers.Dropout(0.5)(a)
age_output = layers.Dense(1, activation='linear', name='age_output')(a)

model = models.Model(inputs=inputs, outputs=[gender_output, age_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'gender_output': 'binary_crossentropy', 'age_output': 'mse'},
    loss_weights={'gender_output': 1.0, 'age_output': 1.0}, # Changed to 1.0 each for balanced training
    metrics={'gender_output': 'accuracy', 'age_output': 'mae'}
)

print("🔄 Loading images... (This might take a while)")
X = []
y_gender = []
y_age = []

# Using ALL available images (or at least 20,000) for better accuracy
limit = min(20000, len(df)) 
for i, row in df.head(limit).iterrows(): 
    processed_img = load_and_preprocess_image(row['image'])
    if processed_img is not None:
        X.append(processed_img)
        y_gender.append(row['gender'])
        y_age.append(row['age'])

X = np.array(X)
y_gender = np.array(y_gender)
y_age = np.array(y_age)

# --- NEW: CALLBACKS FOR BETTER TRAINING ---
# 1. Stop if validation doesn't improve
# 1. Stop if validation doesn't improve
early_stop = callbacks.EarlyStopping(monitor='val_gender_output_accuracy', patience=5, restore_best_weights=True, mode='max')
# 2. Reduce learning rate if it gets stuck
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

print(f"🚀 TRAINING STARTING NOW on {len(X)} images...")
history = model.fit(
    X, 
    {'gender_output': y_gender, 'age_output': y_age}, 
    epochs=50, # Increased epochs
    batch_size=32,
    validation_split=0.2, # 20% validation
    callbacks=[early_stop, reduce_lr]
)

if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/age_gender_model.h5')
print("✅ Success! High-Accuracy Combined Model saved.")