import streamlit as st
from ultralytics import YOLO
import os

# Get the directory of the current script (app.py)
# This ensures it works on both your local machine and the cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'weights', 'best.pt')

# Initialize the model
try:
    model = YOLO(model_path)
    st.success("Model loaded successfully from GitHub!")
except Exception as e:
    st.error(f"Error loading model: {e}")
