import cv2
import sys

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    sys.exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Camera ", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()