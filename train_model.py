import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Loading balanced dataset...")
# Load balanced dataset
df = pd.read_csv('data/mitbih_balanced_train.csv', header=None)

# Separate Features (Signal) and Target
X = df.iloc[:, :187].values
y = df.iloc[:, 187].values

# Reshape X for 1D CNN: (samples, time_steps, features) -> (samples, 187, 1)
X = X.reshape(X.shape[0], 187, 1)

print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

# Build Model (Simple 1D-CNN)
model = models.Sequential([
    # Input Layer
    layers.Input(shape=(187, 1)),
    
    # First Conv Block
    layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
    layers.MaxPool1D(pool_size=2),
    
    # Second Conv Block
    layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
    layers.MaxPool1D(pool_size=2),
    
    # Flatten
    layers.Flatten(),
    
    # Classifier Head
    layers.Dense(64, activation='relu'),
    
    # Output Layer (5 classes)
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("Starting training (Best Model Configuration)...")
# Train the model
history = model.fit(
    X, y,
    epochs=20,  # 20 epochs on balanced data
    validation_split=0.2,
    batch_size=32
)

# Save the model
model_path = 'models/ecg_model_cnn_best.keras'
model.save(model_path)

print(f"Training Complete. Best model saved to {model_path}")
