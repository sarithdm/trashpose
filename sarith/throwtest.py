from ultralytics import YOLO
import numpy as np

# Load the pretrained YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Predict poses from an image
results = model('throw_image.jpg')  # replace 'throw_image.jpg' with the path to your image

# Extract keypoints from the first result
keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to check if the keypoints match an arm extended pose
def is_arm_extended_pose(keypoints):
    if keypoints is None:
        return False

    # Define indices for relevant keypoints
    right_wrist_index = 10
    right_elbow_index = 8
    right_shoulder_index = 6

    # Ensure that the keypoints array has enough keypoints
    if keypoints.shape[0] <= max(right_wrist_index, right_elbow_index, right_shoulder_index):
        return False

    # Extract coordinates for relevant keypoints
    right_wrist = keypoints[right_wrist_index, :2]
    right_elbow = keypoints[right_elbow_index, :2]
    right_shoulder = keypoints[right_shoulder_index, :2]

    # Calculate angles
    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Check relative positions and angles for an extended arm pose
    if (right_wrist[0] > right_elbow[0] and
        right_elbow[0] > right_shoulder[0] and 
        elbow_angle > 160):  # Extended arm typically forms a straight line
        return True

    return False

# Analyze the keypoints data to check for an extended arm pose
extended_arm_detected = is_arm_extended_pose(keypoints)
if extended_arm_detected:
    print("Extended arm pose detected in the image!")
else:
    print("No extended arm pose detected in the image.")
