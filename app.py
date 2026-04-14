import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Aerial Sentinel AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4255; }
    [data-testid="stSidebar"] { background-color: #11141c; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2563/2563412.png", width=80)
    st.title("Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    st.info("Adjust the threshold to filter out low-confidence detections.")
    st.divider()
    st.markdown("### Internship Project\n**Labmentix AI Lab**")

# --- Model Loading Logic ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(current_dir, 'weights', 'best.pt'), os.path.join(current_dir, 'best.pt')]
    for path in paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

model = load_model()

# --- Main Dashboard Header ---
st.title("🚁 Aerial Sentinel: Bird & Drone Detection")
st.markdown("### Intelligent Computer Vision for Airspace Monitoring")
st.divider()

if model is None:
    st.error("⚠️ AI Model file (best.pt) not found. Please verify your repository structure.")
else:
    # --- UI Layout: Upload & Info ---
    col_u1, col_u2 = st.columns([2, 1])
    
    with col_u1:
        uploaded_file = st.file_uploader("Upload an aerial snapshot...", type=["jpg", "jpeg", "png"])
    
    with col_u2:
        st.markdown("#### System Status")
        if uploaded_file:
            st.success("🟢 Ready for Inference")
        else:
            st.warning("⚪ Waiting for Input")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Inference Stage
        start_time = time.time()
        results = model(image, conf=conf_threshold)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000 # Convert to ms
        num_detections = len(results[0].boxes)

        # --- Dashboard Metrics ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Inference Time", f"{inference_time:.1f} ms")
        with m2:
            st.metric("Objects Detected", num_detections)
        with m3:
            st.metric("Model Precision", "YOLOv8s")

        # --- Visual Results ---
        res_plotted = results[0].plot()
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Original Image")
            st.image(image, use_container_width=True)
        with c2:
            st.markdown("##### AI Visual Output")
            st.image(res_plotted, use_container_width=True)
            
        # --- Detection Log ---
        with st.expander("View Detection Meta-Data"):
            st.write(results[0].boxes.data)
