import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image

st.set_page_config(page_title="Aerial Detection", layout="wide")
st.title("🚁 Aerial Bird & Drone Detection")

# Logic to find the model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check two possible locations
path_option1 = os.path.join(current_dir, 'weights', 'best.pt')
path_option2 = os.path.join(current_dir, 'best.pt')

model_path = path_option1 if os.path.exists(path_option1) else path_option2

# Initialize the model
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_model(path):
    return YOLO(path)

if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.sidebar.success("✅ Model Loaded")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Could not find best.pt. Please check your GitHub folder structure.")

# Image Uploader
uploaded_file = st.file_uploader("Upload an aerial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and 'model' in locals():
    image = Image.open(uploaded_file)
    
    # Run Prediction
    with st.spinner('Analyzing image...'):
        results = model(image, conf=0.25) # Run detection
        res_plotted = results[0].plot() # Plot results
        
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(res_plotted, caption="Detected Objects", use_container_width=True)
