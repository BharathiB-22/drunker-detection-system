import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True)

# Eye landmark indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_eye_state(landmarks, w, h):
    # Left eye
    lt = (int(landmarks[LEFT_EYE_TOP].x * w), int(landmarks[LEFT_EYE_TOP].y * h))
    lb = (int(landmarks[LEFT_EYE_BOTTOM].x * w), int(landmarks[LEFT_EYE_BOTTOM].y * h))

    # Right eye
    rt = (int(landmarks[RIGHT_EYE_TOP].x * w), int(landmarks[RIGHT_EYE_TOP].y * h))
    rb = (int(landmarks[RIGHT_EYE_BOTTOM].x * w), int(landmarks[RIGHT_EYE_BOTTOM].y * h))

    left_dist = distance(lt, lb)
    right_dist = distance(rt, rb)

    avg = (left_dist + right_dist) / 2

    if avg < 6:
        return "Closed"
    return "Open"


# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Eye state detection
            eye_state = get_eye_state(landmarks, w, h)

            # Draw eye points
            for idx in [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM]:
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

            # Display result
            cv2.putText(
                frame,
                f"Eyes: {eye_state}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

    cv2.imshow("Face Landmark & Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()