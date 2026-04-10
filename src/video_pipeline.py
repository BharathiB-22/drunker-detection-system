import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Get frame size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

print("Press 'q' to quit")

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

cap.release()
out.release()
cv2.destroyAllWindows()

print("Output saved as output_video.mp4")