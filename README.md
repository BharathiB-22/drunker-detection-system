# 🍷 Drunken Detection System

## 🚀 Project Overview
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

## 📅 Day 4 Progress – Face Analysis Module

### 🎯 Objective
To perform real-time facial analysis using deep learning and landmark-based techniques.

---

### ⚙️ Implementation

- Integrated DeepFace for facial analysis
- Detected dominant emotion in real time
- Estimated age and gender for contextual understanding
- Used threading to improve performance during analysis
- Extracted facial landmarks using MediaPipe Face Mesh
- Implemented Eye Aspect Ratio (EAR) for eye-state detection
- Classified eye state as:
  - Open
  - Closed / Droopy

---

### 🧠 Models Used

- **DeepFace**
  - Used for emotion detection
  - Used for age estimation
  - Used for gender classification

- **MediaPipe Face Mesh**
  - Used for facial landmark detection
  - Provides 468 facial landmarks
  - Used for eye tracking and facial geometry analysis

---

### ⚠️ Environment Setup Note

Two separate virtual environments were used:

- **`venv`**
  - OpenCV
  - DeepFace
  - TensorFlow

- **`venv_landmark`**
  - MediaPipe

**Why two environments were used:**

DeepFace (with TensorFlow) and MediaPipe caused dependency conflicts, mainly related to `protobuf` versions.  
To avoid runtime issues, both modules were tested in separate environments.

This helped us:
- run DeepFace smoothly
- run MediaPipe without version conflicts
- keep both modules stable during development

---

### 📤 Output

- Real-time emotion analysis displayed on screen
- Age and gender estimation shown for context
- Eye-state detection output displayed using landmarks
- Separate working modules created for:
  - DeepFace analysis
  - Landmark-based eye detection

---
## 📅 Day 5 Progress – CNN Model Development

### 🎯 Objective
To build a deep learning model for drunken state classification.

---

### ⚙️ Implementation

- Prepared dataset with two classes:
  - Drunk
  - Normal
- Cleaned the dataset by:
  - removing duplicate images
  - removing invalid/corrupt files
- Split dataset into:
  - Train (70%)
  - Validation (20%)
  - Test (10%)
- Built a custom CNN model using PyTorch
- Used image size of `128 × 128`
- Added convolution, activation, pooling, and fully connected layers
- Used dropout to reduce overfitting

---

### 🧠 Model Used

- **Custom CNN (PyTorch)**
- Chosen because:
  - it is suitable for image classification
  - it is lightweight for a student-level project
  - it is easy to train and explain

---

### 🚀 Training Highlights

- Trained the CNN on drunk and normal image classes
- Learned visual patterns such as:
  - facial appearance
  - eye condition
  - structural facial features
- Best Validation Accuracy achieved: **~97.5%**
- Final Test Accuracy achieved: **~97.7%**

---

### 📤 Output

- Trained model saved to:
  `models/drunk_cnn.pth`

- Classification classes:
  - `Drunk`
  - `Normal`

---

## 📅 Day 6 Progress – Real-Time Classification & Behavioral Analysis

### 🎯 Objective
To integrate the trained CNN model into a real-time system and perform behavioral analysis.

---

### ⚙️ Implementation

- Captured webcam feed using OpenCV
- Detected faces using Haar Cascade
- Selected the largest detected face for better stability
- Applied the trained CNN model for real-time classification
- Displayed prediction on screen as:
  - **DRUNK** (Red)
  - **NORMAL** (Green)

- Added behavioral analysis using MediaPipe landmarks

#### Behavioral cues implemented:
- **Eye State Detection**
  - Used Eye Aspect Ratio (EAR)
  - Detected open or closed/droopy eyes

- **Facial Asymmetry Detection**
  - Compared distances between nose and mouth corners
  - Measured imbalance in facial structure

- **Head Pose Estimation**
  - Calculated angle between eye landmarks
  - Classified head position as:
    - Straight
    - Tilted / Swaying

---

### 🧠 Models Used

- **Haar Cascade Classifier**
  - Used for real-time face detection

- **Custom CNN Model**
  - Used for drunken state classification

- **MediaPipe Face Mesh**
  - Used for behavioral analysis through landmarks

---

### 🚀 Advanced Features

- Combined classification and explainable behavioral cues in one pipeline
- Used facial behavior to support model output interpretation
- Improved system understanding by showing why the person may appear intoxicated

---

### 📤 Output

- Real-time webcam prediction displayed on screen
- Face detected and classified as Drunk / Normal
- Behavioral cues displayed for:
  - eye droopiness
  - facial imbalance
  - head tilt
- Final system works as a real-time drunken detection pipeline

---
### 🔜 Final System Pipeline

```text
Camera Input
   ↓
Face Detection (OpenCV)
   ↓
CNN Classification (Drunk / Normal)
   ↓
Behavioral Analysis
   ├── Eye State
   ├── Facial Asymmetry
   └── Head Pose
   ↓
Display Results
