import cv2
import numpy as np
from ultralytics import YOLO

# Load the pretrained YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Open video file
cap = cv2.VideoCapture('throw_video.mp4')

# List to store keypoint data
keypoints_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict poses
    results = model(frame)

    # Extract keypoints
    for result in results:
        keypoints = result.keypoints.cpu().numpy() if result.keypoints else []
        keypoints_data.append(keypoints)

    # Visualization (optional)
    annotated_frame = results[0].plot()
    cv2.imshow('Pose Estimation', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

