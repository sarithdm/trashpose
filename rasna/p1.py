import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_throw_pose(landmarks):
    """
    Detects a throw pose based on arm extension.
    Args:
        landmarks: List of pose landmarks from MediaPipe.
    Returns:
        bool: True if throwing pose is detected, False otherwise.
    """
    if not landmarks:
        return False

    # Get key landmarks (right arm as an example)
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Calculate the angle between shoulder, elbow, and wrist
    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    angle = calculate_angle(shoulder, elbow, wrist)

    # Throwing pose criteria: Arm angle < 90 degrees (extension)
    return angle < 90

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect throwing pose
        if detect_throw_pose(results.pose_landmarks.landmark):
            cv2.putText(frame, "Throw Pose Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Throw Pose Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
