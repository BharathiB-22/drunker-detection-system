import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Landmark Indices
# Eyes
L_EYE = [33, 160, 158, 133, 153, 144] 
R_EYE = [362, 385, 387, 263, 373, 380]
# Symmetry (Mouth corners and Nose)
MOUTH_L = 61
MOUTH_R = 291
NOSE_TIP = 1

def get_ear(landmarks, eye_indices, w, h):
    # Calculate Eye Aspect Ratio (EAR)
    # Distance between vertical landmarks / distance between horizontal
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * w, lm.y * h]))
    
    # Vertical distances
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    # Horizontal distance
    h_dist = np.linalg.norm(coords[0] - coords[3])
    
    return (v1 + v2) / (2.0 * h_dist)

def get_symmetry(landmarks, w, h):
    # Compare nose-to-left-mouth vs nose-to-right-mouth
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    m_left = np.array([landmarks[MOUTH_L].x * w, landmarks[MOUTH_L].y * h])
    m_right = np.array([landmarks[MOUTH_R].x * w, landmarks[MOUTH_R].y * h])
    
    dist_l = np.linalg.norm(nose - m_left)
    dist_r = np.linalg.norm(nose - m_right)
    
    # Ratio of symmetry (closer to 1.0 is perfectly symmetric)
    return abs(dist_l - dist_r)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        
        # 1. Eye State (EAR)
        ear_l = get_ear(lms, L_EYE, w, h)
        ear_r = get_ear(lms, R_EYE, w, h)
        avg_ear = (ear_l + ear_r) / 2.0
        eye_status = "Closed/Droopy" if avg_ear < 0.22 else "Open"

        # 2. Facial Asymmetry
        asym_score = get_symmetry(lms, w, h)
        asym_status = "High (Slacking)" if asym_score > 12 else "Normal"

        # 3. Head Pose (Angle)
        lx, ly = lms[33].x * w, lms[33].y * h
        rx, ry = lms[263].x * w, lms[263].y * h
        angle = math.degrees(math.atan2(ry - ly, rx - lx))
        pose = "Straight" if abs(angle) < 7 else "Tilted/Swaying"

        # --- DRAWING ---
        color = (0, 255, 0) # Green
        if eye_status == "Closed/Droopy" or asym_status == "High (Slacking)":
            color = (0, 0, 255) # Red Alert

        cv2.putText(frame, f"EAR: {avg_ear:.2f} ({eye_status})", (30, 50), 2, 0.7, color, 2)
        cv2.putText(frame, f"Asymmetry: {asym_score:.1f} ({asym_status})", (30, 80), 2, 0.7, color, 2)
        cv2.putText(frame, f"Head Pose: {angle:.1f} ({pose})", (30, 110), 2, 0.7, color, 2)

    cv2.imshow('Advanced Behavioral Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()