import cv2
import numpy as np
import math

# Function to calculate Euclidean distance
def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Function to detect throwing action
def detect_throw(initial_position, new_position, threshold=50):
    distance = calculate_distance(initial_position, new_position)
    return distance > threshold

# Initialize video capture
video_path = "C:\\Users\\Rasna\\trashpose\\rasna\\best.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Define parameters for object detection (using color-based detection as an example)
lower_color = np.array([0, 120, 70])  # Adjust for your object's color (HSV format)
upper_color = np.array([10, 255, 255])

# Variables to track object positions
initial_position = None
throw_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the object based on color range
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        current_position = (x + w // 2, y + h // 2)  # Object's center

        # Check for throwing action
        if initial_position is None:
            initial_position = current_position  # Set initial position when first detected
        else:
            # Check if the object moved more than the threshold distance (which could indicate a throw)
            if detect_throw(initial_position, current_position):
                throw_detected = True
                print(f"Throw detected! Object moved from {initial_position} to {current_position}")
            else:
                throw_detected = False

            # Update initial position to the current one after checking
            initial_position = current_position

        # Draw bounding box and position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, current_position, 5, (0, 0, 255), -1)

    # Resize the frame to show it in the corner (smaller size)
    frame_resized = cv2.resize(frame, (320, 240))  # Resize to fit in the corner
    cv2.imshow("Video", frame_resized)

    # Display "Throw detected" when a throw is detected
    if throw_detected:
        print("Throw detected!")

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
