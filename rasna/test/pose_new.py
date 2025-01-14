import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Function to detect throwing pose
def detect_throw_pose(landmarks):
    if not landmarks:
        return False

    # Right arm landmarks
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Calculate angles
    arm_angle = calculate_angle(shoulder, elbow, wrist)
    torso_angle = calculate_angle(shoulder, hip, elbow)

    # Additional checks: Wrist forward motion
    wrist_forward = wrist.x > shoulder.x

    # Refined throwing pose criteria
    if arm_angle < 90 and 60 < torso_angle < 120 and wrist_forward:
        return True
    return False

# Start video capture
cap = cv2.VideoCapture(r'C:\Users\Rasna\.vscode\test\pose_vd.mp4')

# Buffer to ensure temporal consistency
pose_buffer = []
buffer_size = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for performance (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Detect throwing pose
        is_throwing = detect_throw_pose(results.pose_landmarks.landmark)
        pose_buffer.append(is_throwing)

        # Maintain buffer size
        if len(pose_buffer) > buffer_size:
            pose_buffer.pop(0)

        # Confirm throwing pose only if detected consistently
        if sum(pose_buffer) >= 3:  # At least 3 out of 5 frames
            cv2.putText(frame, "Throw Pose Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Throw Pose Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
