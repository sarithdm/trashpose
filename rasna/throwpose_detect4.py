import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def detect_throw_pose(landmarks):
    """
    Custom function to detect a throw pose based on landmarks.
    Modify this function based on the specific criteria for a throw pose.
    """
    try:
        # Example: Check the position of right shoulder, elbow, and wrist
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calculate angles
        shoulder_to_elbow = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        elbow_to_wrist = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        angle = np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) / 
            (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist))
        )
        angle = np.degrees(angle)

        # Threshold for detecting throw pose
        if 80 < angle < 120:  # Example range for a throw pose
            return True
    except Exception as e:
        print(f"Error calculating pose: {e}")
    return False

# Open the webcam or video file
cap = cv2.VideoCapture(0)  # Change '0' to a file path for a video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(rgb_frame)

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect the throw pose
        if detect_throw_pose(results.pose_landmarks.landmark):
            cv2.putText(frame, "Throw Pose Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Throw Pose Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
