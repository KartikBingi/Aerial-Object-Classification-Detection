import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Aerial Object Detection", layout="wide")

st.title("🚁 Aerial Bird & Drone Detection System")
st.write("Upload an aerial image to classify and detect objects.")

# Load the model you just trained
# Using the path from your successful run
model_path = r'runs\detect\aerial_model_v13\weights\best.pt'
model = YOLO(model_path)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Run Detection
    with st.spinner('Detecting...'):
        results = model.predict(image, conf=0.25)
        
        # Draw the boxes on the image
        res_plotted = results[0].plot()
        
    with col2:
        st.image(res_plotted, caption="Detection Results", use_container_width=True)
        
    # Show statistics
    st.subheader("Detection Results Summary")
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            st.info("No objects detected.")
        else:
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                st.write(f"- Found **{label}** with **{conf:.2%}** confidence.")