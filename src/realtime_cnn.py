import cv2
import torch
import torch.nn as nn
import threading
import numpy as np
from collections import deque
from torchvision import transforms
from deepface import DeepFace

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- LOAD MODEL ----------------
model = SimpleCNN().to(DEVICE)
try:
    model.load_state_dict(torch.load("models/drunk_cnn.pth", map_location=DEVICE))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model weights: {e}")
model.eval()

# ---------------- CNN TRANSFORM ----------------
# Added Normalization: standard for PyTorch pre-trained styles
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ["drunk", "normal"]

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- SHARED STATE ----------------
emotion_lock = threading.Lock()
last_emotion = "Analyzing..."
last_conf = 0.0
emotion_running = False

# Smoothing buffer: stores last 10 predictions to prevent flickering
prediction_buffer = deque(maxlen=10)

def analyze_emotion(face_crop):
    global last_emotion, last_conf, emotion_running
    try:
        result = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion", "unknown")
        conf = float(result.get("emotion", {}).get(emotion, 0.0))

        with emotion_lock:
            last_emotion = emotion
            last_conf = conf
    except:
        pass
    finally:
        emotion_running = False

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)
frame_idx = 0
ANALYZE_EVERY = 20 # Increased slightly for better performance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    if len(faces) > 0:
        # Keep only largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[y:y + h, x:x + w]

        # 1. CNN Drunk Prediction
        try:
            input_tensor = transform(face_crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_tensor)
                # Apply Softmax to see if the model is actually confident
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf_val, pred = torch.max(probs, 1)
                
                # Only add to buffer if confidence is high enough
                if conf_val.item() > 0.6:
                    prediction_buffer.append(classes[pred.item()])
        except Exception as e:
            print(f"Inference error: {e}")

        # Determine smoothed label
        if len(prediction_buffer) > 0:
            cnn_label = max(set(prediction_buffer), key=prediction_buffer.count)
        else:
            cnn_label = "Scanning..."

        # 2. Emotion Threading
        if frame_idx % ANALYZE_EVERY == 0 and not emotion_running:
            emotion_running = True
            threading.Thread(
                target=analyze_emotion,
                args=(face_crop.copy(),),
                daemon=True
            ).start()

        # 3. UI Drawing
        color = (0, 0, 255) if cnn_label == "drunk" else (0, 255, 0)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        # Label background for readability
        cv2.rectangle(display, (x, y - 60), (x + w, y), color, -1)
        
        with emotion_lock:
            emotion_text = f"Emotion: {last_emotion} ({last_conf:.1f}%)"

        cv2.putText(display, f"STATE: {cnn_label.upper()}", (x + 5, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, emotion_text, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detection System", display)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()