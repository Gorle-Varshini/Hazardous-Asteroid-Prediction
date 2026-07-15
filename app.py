import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Helper: find artifact by common names or inside misnamed directories
def find_artifact(candidates):
    for name in candidates:
        p = Path(name)
        if p.exists() and p.is_file():
            return p
    # also allow case where a directory was created by mistake (e.g. asteroid_model.pk1/)
    for name in candidates:
        d = Path(name)
        if d.exists() and d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    return f
    return None


# Attempt to locate model and scaler files (accept common misnaming)
model_file = find_artifact(["asteroid_model.pkl", "asteroid_model.pk1", "asteroid_model"])
scaler_file = find_artifact(["scaler.pkl", "scaler.pk1", "scaler"])

model = None
scaler = None

if model_file is None:
    st.warning("Model file not found (expected asteroid_model.pkl). Using placeholder until fixed.")
else:
    try:
        model = joblib.load(model_file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if scaler_file is None:
    st.warning("Scaler file not found (expected scaler.pkl). Using placeholder until fixed.")
else:
    try:
        scaler = joblib.load(scaler_file)
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")

# Page title
st.title("🚀 Hazardous Asteroid Prediction AI")

st.write("Enter the asteroid details below:")

# User Inputs
est_diameter_min = st.number_input("Estimated Diameter Minimum", value=0.0)
est_diameter_max = st.number_input("Estimated Diameter Maximum", value=0.0)
relative_velocity = st.number_input("Relative Velocity", value=0.0)
miss_distance = st.number_input("Miss Distance", value=0.0)
orbiting_body = st.number_input("Orbiting Body (Encoded)", value=0)
sentry_object = st.number_input("Sentry Object (Encoded)", value=0)
absolute_magnitude = st.number_input("Absolute Magnitude", value=0.0)

# Prediction
if st.button("Predict"):

    input_data = np.array([[
        est_diameter_min,
        est_diameter_max,
        relative_velocity,
        miss_distance,
        orbiting_body,
        sentry_object,
        absolute_magnitude
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Hazardous Asteroid")
    else:
        st.success("✅ Non-Hazardous Asteroid")