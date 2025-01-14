from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best.pt')  # Replace with the path to your trained model

# Open the video source (e.g., webcam or video file)
video_source = 0  # Use 0 for webcam or replace with a video file path
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame or video ended.")
        break

    # Perform pose detection
    results = model(frame)

    # Visualize results
    annotated_frame = results[0].plot()  # Plot the detections on the frame

    # Display the frame
    cv2.imshow('Throw Pose Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()