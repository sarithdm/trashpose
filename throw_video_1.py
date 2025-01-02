import cv2
import numpy as np
from ultralytics import YOLO

# Load the pretrained YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Open video file
cap = cv2.VideoCapture('throw_video_yes.mp4')

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
        if result.keypoints is not None:
            keypoints = result.keypoints.cpu().numpy()
            keypoints_data.append(keypoints)

    # Visualization (optional)
    annotated_frame = results[0].plot()  # Plot the first result
    cv2.imshow('Pose Estimation', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Example function to check if keypoints follow a throwing pattern
def is_throwing(keypoints_sequence):
    # Define your throwing action pattern here
    # For simplicity, let's assume we check the right wrist movement
    right_wrist_index = 10  # Index of the right wrist keypoint
    movements = [kp[right_wrist_index, :2] for kp in keypoints_sequence if kp.shape[0] > right_wrist_index]
    if len(movements) < 2:
        return False
    # Apply logic to determine if the movements match a throwing action
    distances = [np.linalg.norm(movements[i] - movements[i-1]) for i in range(1, len(movements))]
    if max(distances) > 5:  # Define a threshold for movement
        return True
    return False

# Analyze the keypoints data
throwing_detected = is_throwing(keypoints_data)
if throwing_detected:
    print("Throwing action detected!")
else:
    print("No throwing action detected.")