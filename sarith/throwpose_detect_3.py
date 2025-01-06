import cv2
from ultralytics import YOLO
import numpy as np
import logging
from collections import deque
from datetime import datetime

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

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Initialize tracking variables
object_positions = deque(maxlen=30)  # Store the last 30 positions
human_positions = deque(maxlen=30)
trash_positions = deque(maxlen=30)

# Start video capture from file
video_path = 'throw_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict poses from the frame
    results = model(frame)
    #print(results)

    # Extract keypoints from the first result
    if results and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        #print(keypoints)
        
        # Ensure there are enough keypoints detected
        if len(keypoints) > 1:
            # Detect throwing pose
            #print(keypoints)
            if is_throwing_pose(keypoints):
                print("Throwing Pose Detected!")

            # Track positions
            human_positions.append(keypoints[0])  # Assuming keypoints[0] is the human position
            #print(human_positions)
            trash_positions.append(keypoints[1])  # Assuming keypoints[1] is the trash position
            #print(trash_positions)
            # Calculate distances and check for illegal dumping
            if len(human_positions) > 1 and len(trash_positions) > 1:
                #print("Human")
                #print(human_positions)
                #print("Trash")
                #print(trash_positions)
                initial_distance = calculate_distance(human_positions[0], trash_positions[0])
                print("Initial Distance:",initial_distance)
                final_distance = calculate_distance(human_positions[-1], trash_positions[-1])
                print("Final Distance:",final_distance)
                if initial_distance < final_distance:  # Example thresholds
                    print("Illegal dumping detected!")
                    # Alert relevant authorities (e.g., send an email or notification)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()