import cv2
import sys

def start_video_stream(camera_index=0, width=640, height=480, fps=30):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {camera_index}")
        return False

    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print("[INFO] Camera started successfully")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            cv2.imshow("Drunken Detection - Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting video stream")
                break

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


if __name__ == "__main__":
    camera_index = 0

    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("[ERROR] Camera index must be an integer")
            sys.exit(1)

    success = start_video_stream(camera_index)
    sys.exit(0 if success else 1)