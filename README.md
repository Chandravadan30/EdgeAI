# EdgeAI: Real-Time Object & Anomaly Detection on Jetson Orin Nano

## Overview
EdgeAI is a real-time object and anomaly detection system built on the **NVIDIA Jetson Orin Nano Developer Kit** using Docker.

The system processes live camera input, detects objects using deep learning, and identifies anomalies based on unexpected object classes. It displays results through a futuristic HUD-style dashboard.

---

##  Features
- Real-time live camera processing
- Object detection using SSD-Mobilenet (Jetson Inference)
- Anomaly detection based on object classes
- Futuristic HUD-style UI dashboard
- Confidence & risk visualization
- Event logging (CSV)
- Screenshot capture
- Docker-based deployment (Jetson optimized)

---

## Technologies Used
- Python
- OpenCV
- Jetson Inference (detectNet)
- Docker
- NVIDIA Jetson Orin Nano

---

## Docker Image Used
dustynv/jetson-inference:r35.4.1

## Run the Project
python3 anomaly_detection.py --source 0



