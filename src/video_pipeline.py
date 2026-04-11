import cv2
import sys
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    sys.exit(1)

# Get frame size and FPS from camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0  # Fallback if camera doesn't report FPS

# Set output path
output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output", "videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_video.mp4")

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("Error: Cannot create output video file")
    cap.release()
    sys.exit(1)

print(f"Recording at {frame_width}x{frame_height}, {fps:.1f} FPS")
print(f"Output: {output_path}")
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Processing (simple overlay)
        cv2.putText(
            frame,
            "Video Pipeline Running",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show video
        cv2.imshow("Live Camera", frame)

        # Save video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Output saved to {output_path}")