import cv2
from ultralytics import YOLO
import numpy as np
import logging

# Suppress YOLO logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Load the pretrained YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Function to detect throwing pose
def is_throwing_pose(keypoints):
    if len(keypoints) < 11:
        return False  # Not enough keypoints detected

    # Define key landmarks
    left_wrist = keypoints[9]  # Assuming keypoints index for left wrist
    left_elbow = keypoints[7]  # Assuming keypoints index for left elbow
    left_shoulder = keypoints[5]  # Assuming keypoints index for left shoulder

    right_wrist = keypoints[10]  # Assuming keypoints index for right wrist
    right_elbow = keypoints[8]  # Assuming keypoints index for right elbow
    right_shoulder = keypoints[6]  # Assuming keypoints index for right shoulder

    # Example heuristic: Check if the wrist is above the shoulder
    # and elbow angle suggests a throwing position
    def is_arm_throwing(wrist, elbow, shoulder):
        return (
            wrist[1] < shoulder[1] and  # Wrist above shoulder
            elbow[1] > shoulder[1] and  # Elbow below shoulder
            wrist[0] > elbow[0]         # Wrist in forward motion
        )

    # Check for left or right arm throwing pose
    left_throwing = is_arm_throwing(left_wrist, left_elbow, left_shoulder)
    right_throwing = is_arm_throwing(right_wrist, right_elbow, right_shoulder)

    return left_throwing or right_throwing

# Start video capture from file
video_path = 'NVR_ch1_main_20241118083054_20241118083251_camera2.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict poses from the frame
    results = model(frame)

    # Extract keypoints from the first result
    if results and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        # Detect throwing pose
        if is_throwing_pose(keypoints):
            print("Throwing Pose Detected!")

cap.release()
cv2.destroyAllWindows()