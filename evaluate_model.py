import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Constants
MODEL_PATH = 'models/ecg_model_cnn_best.keras'
TEST_DATA_PATH = 'data/mitbih_test.csv'

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. PLease train the model first.")
        return

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}.")
        return

    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Loading test dataset...")
    # Load test data (no header)
    df_test = pd.read_csv(TEST_DATA_PATH, header=None)

    # Separate Features and Target
    X_test = df_test.iloc[:, :187].values
    y_test = df_test.iloc[:, 187].values.astype(int)

    # Reshape X_test for the CNN: (samples, time_steps, features)
    X_test = X_test.reshape(X_test.shape[0], 187, 1)

    print(f"Test Data Loaded. Shape: {X_test.shape}")

    print("Running predictions...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("\n--- Classification Report ---")
    class_names = [
        "Normal", 
        "Supraventricular", 
        "Premature Ventricular", 
        "Fusion", 
        "Unclassifiable"
    ]
    report = classification_report(y_test, y_pred_classes, target_names=class_names)
    print(report)

    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
