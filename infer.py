import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# === Load model, scaler, and label encoder ===
model = load_model("models/motion_model.keras", custom_objects={'AttentionLayer': AttentionLayer})
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# === Function to clean and preprocess one CSV file ===
def infer(filepath, features, window_size=100, stride=50):
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Drop NaNs
    df = df.dropna()
    
    # Select only features
    df_features = df[features]

    # Scale
    df_scaled = scaler.transform(df_features)

    # Repackage into DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    # Create sequences
    sequences = []
    for start in range(0, len(df_scaled) - window_size + 1, stride):
        end = start + window_size
        seq = df_scaled.iloc[start:end].values
        sequences.append(seq)
    
    return np.array(sequences)

# === Define your feature columns ===
features = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

# === Choose a file to predict ===
file_to_predict = "data/humans/sensor_data S aysel.csv" # <-- change this path to test other files

# === Preprocess ===
X_input = infer(file_to_predict, features)

# === Predict ===
predictions = model.predict(X_input)
predicted_classes = np.argmax(predictions, axis=1)
decoded_labels = label_encoder.inverse_transform(predicted_classes)

# === Output the results ===
from collections import Counter
most_common = Counter(decoded_labels).most_common(1)[0][0]
print(f"\nPredicted movement: {most_common}")
