import cv2
import sys
import os
import logging
import threading
from deepface import DeepFace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)


def run_face_emotion(camera_index=0, analyze_every=15):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output", "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "face_emotion_output.mp4")

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        logger.error("Cannot create output video")
        cap.release()
        sys.exit(1)

    logger.info(f"Recording at {w}x{h}, {fps:.1f} FPS | Output: {out_path}")
    logger.info("Press 'q' to quit")

    result_lock = threading.Lock()
    last_box = None
    last_emotion = "Analyzing..."
    last_conf = 0.0
    last_age = 0
    last_gender = "Unknown"
    analyzing = False

    def analyze(frame):
        nonlocal last_box, last_emotion, last_conf, last_age, last_gender, analyzing
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            if len(faces) == 0:
                with result_lock:
                    last_box = None
                    last_emotion = "No Face"
                    last_conf = 0.0
                    last_age = 0
                    last_gender = "Unknown"
                analyzing = False
                return

            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, bw, bh = face

            face_crop = frame[y:y + bh, x:x + bw]

            res = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion", "age", "gender"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )

            res = res[0] if isinstance(res, list) else res

            emotion = res.get("dominant_emotion", "unknown")
            conf = res.get("emotion", {}).get(emotion, 0.0)
            age = res.get("age", 0)
            gender = res.get("dominant_gender", "Unknown")

            if gender == "Man":
                gender = "Male"
            elif gender == "Woman":
                gender = "Female"

            with result_lock:
                last_box = (x, y, bw, bh)
                last_emotion = emotion
                last_conf = conf
                last_age = age
                last_gender = gender

        except Exception as e:
            logger.warning(f"Error: {e}")

        analyzing = False

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % analyze_every == 0 and not analyzing:
                analyzing = True
                threading.Thread(target=analyze, args=(frame.copy(),), daemon=True).start()

            display = frame.copy()

            with result_lock:
                if last_box:
                    x, y, bw, bh = last_box

                    cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                    cv2.putText(
                        display,
                        f"{last_emotion} ({last_conf:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        display,
                        f"Age: {last_age}",
                        (x, y + bh + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        display,
                        f"Gender: {last_gender}",
                        (x, y + bh + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

            cv2.imshow("Face Emotion Detection", display)
            out.write(display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    run_face_emotion()