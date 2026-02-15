import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import tensorflow as tf
from tensorflow import keras

# Set page configuration
st.set_page_config(page_title="ECG Arrhythmia Explorer", layout="wide")

# Constants
MODEL_PATH = "models/ecg_model_cnn_best.keras"

# Mapping for Labels (MIT-BIH Standard)
LABEL_MAP = {
    0: "Normal Beat",
    1: "Supraventricular premature beat",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal beat",
    4: "Unclassifiable beat"
}

@st.cache_resource
def load_keras_model():
    """
    Load the trained Keras model.
    Uses cache_resource to load the model only once.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error reading model file: {e}")
        return None

def main():
    st.title("ECG Arrhythmia Explorer")
    st.markdown("### Upload your ECG data for automated arrhythmia detection.")

    # Load Model
    model = load_keras_model()

    if model is None:
        st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the model is trained and saved.")
        return

    # File Uploader
    uploaded_file = st.file_uploader("Upload your ECG CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file with 188 columns (features + label). The app will analyze each heartbeat row.")
        st.markdown("""
        **Format Requirements:**
        - CSV format
        - **No header** row
        - 187 columns for signal data
        - 1 column for label (optional, but expected by current logic for visualization consistency)
        """)
        return

    # Process Uploaded File
    try:
        df = pd.read_csv(uploaded_file, header=None)
        st.success(f"File uploaded successfully! Analyzed {len(df)} heartbeats.")
        st.divider()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Analysis Loop
    # We'll limit to first 10 rows for demo purposes if the file is huge, 
    # but requirement said "Loop through each row", so we'll do paginated or just all.
    # Let's show all but maybe add a stop button if it's huge? 
    # For now, let's assume reasonable file size or just iterate.
    
    # Check dimensions
    if df.shape[1] < 187:
        st.error("Error: CSV must have at least 187 columns for signal data.")
        return

    for index, row in df.iterrows():
        # Layout for each heartbeat
        st.subheader(f"Heartbeat #{index + 1}")
        
        # Extract Signal
        signal_values = row[0:187]
        signal_array = signal_values.values
        
        # Reshape for Model: (1, 187, 1)
        input_data = signal_array.astype(np.float32).reshape(1, 187, 1)
        
        # Predict
        predictions = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence_score = np.max(predictions) * 100
        predicted_diagnosis = LABEL_MAP.get(predicted_class, "Unknown Class")
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### AI Diagnosis")
            if predicted_class == 0:
                st.success(f"**{predicted_diagnosis}**")
            else:
                st.error(f"**{predicted_diagnosis}**")
            
            st.metric("Confidence", f"{confidence_score:.2f}%")
            
            # If label exists (column 187), show it for reference
            if df.shape[1] > 187:
                actual_label_code = int(row[187])
                actual_name = LABEL_MAP.get(actual_label_code, "Unknown")
                st.caption(f"Ground Truth (from file): {actual_name}")

        with col2:
            # Plot
            plot_df = pd.DataFrame({
                "Time": range(len(signal_values)),
                "Amplitude": signal_values.values
            })
            
            fig = px.line(
                plot_df, 
                x="Time", 
                y="Amplitude",
                height=250,
                template="plotly_white"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            fig.update_traces(line_color='#d62728', line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            
        st.divider()

if __name__ == "__main__":
    main()
