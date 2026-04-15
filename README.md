# 🍷 Drunken Detection System

## 📌 Project Overview
This project aims to develop an AI-based real-time drunken detection system using computer vision and deep learning techniques. The system analyzes facial features and behavioral patterns from video streams to determine whether a person is in a drunken state.

---

## 🧠 Tech Stack
- Python 3.11
- OpenCV
- DeepFace
- YOLOv8
- PyTorch
- Node.js (planned)
- React (planned)
- MongoDB (planned)
- Qwen CLI (AI code generation tool)

---

## ⚙️ Day 1 Progress

### ✅ Environment Setup
- Installed Python 3.11
- Created virtual environment
- Installed OpenCV library

### ✅ Qwen CLI Integration
- Installed Node.js
- Installed npm
- Installed Qwen CLI
- Logged in via browser authentication
- Integrated Qwen CLI with VS Code

### ✅ GitHub Setup
- Created repository
- Added project structure
- Uploaded initial code

### ✅ OpenCV Video Pipeline
- Implemented real-time webcam capture
- Displayed live video feed
- Added error handling and clean exit

---

## 🎥 Camera Test Module

File: `src/camera_test.py`

### ✔ Functionality:
- Captures live video from webcam
- Displays real-time frames
- Handles errors gracefully
- Allows exit using 'q' key

### ▶️ How to Run

```bash
python src/camera_test.py
<<<<<<< HEAD

=======
```
---

## ⚙️ Day 2 Progress

### 🎯 Objective
To enhance the basic video pipeline by adding frame processing and video recording functionality.

---

### ✅ Video Processing Pipeline

- Captured real-time video frames using OpenCV
- Retrieved camera properties such as resolution and FPS
- Implemented fallback FPS handling for unsupported devices
- Added frame-level processing (text overlay)
- Displayed processed frames in real-time

---

### ✅ Video Recording System

- Created structured output directory:
  `data/output/videos/`
- Used `cv2.VideoWriter` to record video
- Maintained consistency between input and output resolution
- Saved processed frames into an `.mp4` file

---

## 🎥 Video Pipeline Module

File: `src/video_pipeline.py`

### ✔ Functionality:
- Captures live video from webcam
- Applies basic processing on each frame
- Displays video in real-time
- Records and saves video output
- Allows exit using 'q' key

### ▶️ How to Run

```bash
python src/video_pipeline.py
```
---

## 📅 Day 3 Progress – Face Detection Module

### 🎯 Objective
To implement real-time face detection using computer vision techniques.

---

### ⚙️ Implementation

- Integrated OpenCV Haar Cascade for face detection
- Captured real-time video frames from webcam
- Converted frames to grayscale for efficient processing
- Detected faces using `detectMultiScale()`
- Drew bounding boxes around detected faces
- Displayed face count on screen

---

### 🧠 Model Used

- Haar Cascade Classifier (OpenCV)
- Chosen for its speed and suitability for real-time applications

---

### 🚀 Advanced Features

- Implemented detection optimization (detect every N frames)
- Added face tracking and stabilization to reduce flickering
- Integrated logging system for better debugging
- Implemented graceful shutdown using signal handling

---

### 📤 Output

- Real-time face detection displayed on screen
- Bounding boxes drawn around faces
- Face count displayed dynamically
- Processed video saved to:
  `data/output/videos/face_detection_output.mp4`

---

### 🧠 Understanding

- Learned how face detection works using Haar Cascades
- Understood real-time frame processing
- Learned performance optimization techniques
- Understood importance of stabilizing detection results

---

### 🔜 Next Step
Integrate DeepFace for emotion detection (Day 4)
