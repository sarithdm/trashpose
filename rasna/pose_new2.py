import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Motion history
motion_history = []

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.fff
    Args:
        a, b, c: Points with x and y attributes.
    Returns:
        Angle in degrees.
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def detect_throw_pose(landmarks):
    """
    Detects a static throw pose based on arm and torso angles.
    Args:
        landmarks: List of pose landmarks from MediaPipe.
    Returns:
        bool: True if a throw pose is detected, False otherwise.
    """
    if not landmarks:
        return False

    # Get key landmarks
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Calculate angles
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    torso_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)

    # Static throw pose criteria
    is_right_throw = right_arm_angle < 90 and torso_angle > 20
    is_left_throw = left_arm_angle < 90 and torso_angle > 20

    return is_right_throw or is_left_throw

def detect_throw_sequence(landmarks):
    """
    Detects a throwing motion sequence based on wrist movement over time.
    Args:
        landmarks: List of pose landmarks from MediaPipe.
    Returns:
        bool: True if a throwing motion is detected, False otherwise.
    """
    global motion_history

    # Track right wrist motion
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    motion_history.append(right_wrist.x)

    # Limit motion history size
    if len(motion_history) > 10:
        motion_history = motion_history[-10:]

    # Detect backward and forward motion
    if len(motion_history) >= 5:
        backward_motion = motion_history[-5] < motion_history[-3] < motion_history[-1]
        forward_motion = motion_history[-1] > motion_history[-3] > motion_history[-5]
        return backward_motion or forward_motion

    return False

# Start video capture
cap = cv2.VideoCapture('C:\\Users\\Rasna\\.vscode\\test\\pose_vd.mp4')

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
        landmarks = results.pose_landmarks.landmark

        # Detect throwing pose or sequence
        if detect_throw_pose(landmarks) or detect_throw_sequence(landmarks):
            cv2.putText(frame, "Throw Pose Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Throw Pose Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
