import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    angle = math.acos(dot_product / (magnitude_ab * magnitude_bc + 1e-6))
    return math.degrees(angle)

# Function to detect throwing pose based on angles
def is_throwing_pose(landmarks):
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    shoulder_angle = calculate_angle(right_elbow, right_shoulder, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

    is_elbow_in_throwing_position = 90 <= elbow_angle <= 140
    is_shoulder_in_throwing_position = 70 <= shoulder_angle <= 110

    return is_elbow_in_throwing_position and is_shoulder_in_throwing_position

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if is_throwing_pose(result.pose_landmarks.landmark):
            cv2.putText(frame, "Throwing Pose Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Throwing Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
