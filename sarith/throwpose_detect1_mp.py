import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to detect throwing pose
def is_throwing_pose(landmarks):
    # Define key landmarks
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Example heuristic: Check if the wrist is above the shoulder
    # and elbow angle suggests a throwing position
    def is_arm_throwing(wrist, elbow, shoulder):
        return (
            wrist.y < shoulder.y and  # Wrist above shoulder
            elbow.y > shoulder.y and  # Elbow below shoulder
            wrist.x > elbow.x         # Wrist in forward motion
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

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # Check for pose landmarks
    if result.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Detect throwing pose
        if is_throwing_pose(result.pose_landmarks.landmark):
            cv2.putText(frame, "Throwing Pose Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Throwing Pose Detected!")

    # Display the frame
    cv2.imshow('Throwing Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()