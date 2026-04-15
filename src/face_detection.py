import cv2
import sys
import os
import logging
import signal
import math

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- PARAMETERS ----------------
DETECT_SCALE_FACTOR = 1.1
DETECT_MIN_NEIGHBORS = 5
DETECT_MIN_SIZE = (30, 30)
DETECT_EVERY_N_FRAMES = 3  # optimize CPU

TRACK_DISTANCE_THRESHOLD = 100

# ---------------- TRACKING ----------------
_prev_faces = []

def _centroid(rect):
    x, y, w, h = rect
    return (x + w / 2, y + h / 2)

def _match_faces(current_faces, prev_faces):
    matches = [None] * len(current_faces)
    used_prev = set()

    for i, cur in enumerate(current_faces):
        c_cur = _centroid(cur)
        best_dist = TRACK_DISTANCE_THRESHOLD
        best_idx = None

        for j, prev in enumerate(prev_faces):
            if j in used_prev:
                continue

            dist = math.hypot(
                c_cur[0] - _centroid(prev)[0],
                c_cur[1] - _centroid(prev)[1],
            )

            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx is not None:
            matches[i] = prev_faces[best_idx]
            used_prev.add(best_idx)

    return matches

def _stabilise_faces(current_faces, prev_faces, alpha=0.6):
    matches = _match_faces(current_faces, prev_faces)
    stabilised = []

    for cur, prev in zip(current_faces, matches):
        if prev is not None:
            x = int(alpha * cur[0] + (1 - alpha) * prev[0])
            y = int(alpha * cur[1] + (1 - alpha) * prev[1])
            w = int(alpha * cur[2] + (1 - alpha) * prev[2])
            h = int(alpha * cur[3] + (1 - alpha) * prev[3])
            stabilised.append((x, y, w, h))
        else:
            stabilised.append(cur)

    return stabilised

# ---------------- SIGNAL HANDLING ----------------
_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logger.error("Cannot open camera")
    sys.exit(1)

# ---------------- FACE MODEL ----------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    logger.error("Failed to load Haar Cascade")
    cap.release()
    sys.exit(1)

# ---------------- VIDEO SETTINGS ----------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

# ---------------- OUTPUT PATH ----------------
output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output", "videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "face_detection_output.mp4")

# ---------------- VIDEO WRITER ----------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    logger.error("Cannot create output video")
    cap.release()
    sys.exit(1)

logger.info(f"Recording at {frame_width}x{frame_height}, {fps:.1f} FPS")
logger.info(f"Output: {output_path}")
logger.info("Press 'q' to quit")

# ---------------- MAIN LOOP ----------------
frame_idx = 0

try:
    while not _shutdown_requested:
        ret, frame = cap.read()

        if not ret:
            logger.warning("Failed to read frame")
            break

        # Default: reuse previous detections so boxes stay visible between runs
        faces = list(_prev_faces)

        # Detect every N frames
        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=DETECT_SCALE_FACTOR,
                minNeighbors=DETECT_MIN_NEIGHBORS,
                minSize=DETECT_MIN_SIZE,
            )

            raw_faces = [tuple(f) for f in raw_faces]

            if _prev_faces:
                faces = _stabilise_faces(raw_faces, _prev_faces)
            else:
                faces = raw_faces

            _prev_faces = faces

        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # Show
        cv2.imshow("Face Detection", frame)

        # Save
        out.write(frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"Saved to {output_path}")

