<img width="901" height="357" alt="image" src="https://github.com/user-attachments/assets/b3692818-d4ca-44e9-bdc1-ffcf572864c0" /># Aerial-Object-Classification-Detection
Developed during my Data Science Internship at Labmentix, this project implements a real-time computer vision system capable of distinguishing between Birds and Drones in aerial footage. The system is designed for security surveillance and wildlife monitoring, utilizing the state-of-the-art YOLOv8 architecture.

Project Overview
Developed as part of my Data Science Internship at Labmentix, this project implements an end-to-end computer vision pipeline to solve the "Small Object Detection" problem in aerial surveillance. The system accurately classifies and localizes Birds and Drones in high-altitude imagery, providing a critical tool for airspace security and wildlife monitoring.

Live Demo

https://aerial-object-classification-detection-kipbdwqg5g8x5s6eu4eksg.streamlit.app/

Performance & Metrics
The model was trained for 25 epochs on a diverse aerial dataset, achieving high reliability for security-critical detection.

Metric                                    Value
Mean Average Precision (mAP50)	          0.7654
Inference Time (Avg)                     	3.9ms - 10.2ms
Model Size                                YOLOv8 Small (22.5 MB)
Hardware Used                             NVIDIA GeForce GTX 1650 Ti

Key Technical Challenges & Solutions
1. Hardware-Specific Optimization
During training on the GTX 1650 Ti, initial checks for Automatic Mixed Precision (AMP) failed. To ensure numerical stability and prevent "NaN" loss values, I manually configured the training pipeline to utilize full FP32 precision, resulting in a stable and high-performing weight file (best.pt).

2. Cloud-Native Deployment (Linux)
Transitioning from Windows to a Linux-based Streamlit Cloud environment presented dependency hurdles.
Problem: Missing system-level graphics libraries (libGL.so.1).
Solution: Implemented a packages.txt manifest to bridge the gap between Python code and OS-level requirements.

3. Intelligent Path Handling
Engineered a flexible path-resolution logic in app.py that automatically detects whether the app is running in a local Windows environment or a Linux cloud container, ensuring zero-configuration deployment.
