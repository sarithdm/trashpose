import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 pose model
model = YOLO("yolov8-pose.pt")  # Ensure you have the pose model

def is_throw_pose(keypoints):
    """
    Check if the detected keypoints match a "throw pose".
    Adjust conditions based on your requirements.
    """
    try:
        # Example keypoints indices for YOLOv8 pose:
        # 0: Nose, 1: Left Eye, 2: Right Eye, 5: Left Shoulder, 6: Right Shoulder
        # 7: Left Elbow, 8: Right Elbow, 9: Left Wrist, 10: Right Wrist
        
        # Extract keypoints
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        # Example conditions for throw pose
        if (left_wrist[1] < left_elbow[1] < left_shoulder[1] or
            right_wrist[1] < right_elbow[1] < right_shoulder[1]):
            return True
    except IndexError:
        pass
    return False

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Replace 0 with video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect poses
    results = model.predict(frame, save=False, save_txt=False)
    annotated_frame = results[0].plot()

    for person in results[0].keypoints:
        keypoints = person.xy.cpu().numpy()  # Convert keypoints to numpy array
        if is_throw_pose(keypoints):
            cv2.putText(annotated_frame, "Throw Pose Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Throw Pose Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
