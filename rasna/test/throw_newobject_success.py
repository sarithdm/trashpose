import cv2
import numpy as np

# Initialize video capture
video_path = "C:\\Users\\Rasna\\.vscode\\test\\pose_vd.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Variables for tracking throw action
throw_detected = False
frame_count = 0
new_object_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 500:
            continue

        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect new object appearing in the background
        if not throw_detected and frame_count > 30:
            # Consider a new object detected if it appears stationary in the background
            new_object_detected = True
            print(f"New object detected at position: ({x}, {y}, {w}, {h})")

    # If a new object is detected and a throw action is assumed
    if new_object_detected:
        throw_detected = True
        print("Throw action detected!")

    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground Mask", fgmask)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
