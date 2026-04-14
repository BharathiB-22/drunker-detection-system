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

⚙️ Day 2 – Video Processing & Recording
📌 Objective

Enhance pipeline by adding:

Frame processing
Video recording
Output storage
🔄 Improvements
Added frame-level processing
Implemented video recording
Created structured output folder
Prepared pipeline for AI modules
⚙️ Implementation
📷 Camera Handling
Captured:
Width, Height
FPS
Used fallback FPS = 30
📁 Output Management
Created folder:
data/output/videos/
Used os.makedirs() to avoid errors
🎬 Video Recording
Used cv2.VideoWriter
Codec: mp4v
Saved as .mp4
Maintained same resolution
🧩 Frame Processing
Added overlay text:
Video Pipeline Running
Simulates real-time processing
🖥️ Display
Live video display continues
Exit using 'q' key
📤 Output
Live processed webcam feed
Saved video:
data/output/videos/output_video.mp4
🧠 Learning
Video recording using frames
FPS synchronization
File & directory handling
Pipeline structure for AI integration
⚠️ Challenges
FPS inconsistency
Video encoding formats
File path handling
🔗 System Flow
Camera → Capture → Process → Display → Save
✅ Conclusion
Day 1: Basic webcam pipeline
Day 2: Full video processing system
