import cv2
from ultralytics import YOLO
import os

# Load the trained YOLOv8 model
model = YOLO('best_50_epoch_10_12_2024.pt')  # Adjust the path to your trained model

# Define the classes of interest
classes_of_interest = [
    'Aluminium foil', 'Battery', 'Aluminium blister pack', 'Carded blister pack', 'Other plastic bottle', 
    'Clear plastic bottle', 'Glass bottle', 'Plastic bottle cap', 'Metal bottle cap', 'Broken glass', 
    'Food Can', 'Aerosol', 'Drink can', 'Toilet tube', 'Other carton', 'Egg carton', 'Drink carton', 
    'Corrugated carton', 'Meal carton', 'Pizza box', 'Paper cup', 'Disposable plastic cup', 'Foam cup', 
    'Glass cup', 'Other plastic cup', 'Food waste', 'Glass jar', 'Plastic lid', 'Metal lid', 'Other plastic', 
    'Magazine paper', 'Tissues', 'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastified paper bag', 
    'Plastic film', 'Six pack rings', 'Garbage bag', 'Other plastic wrapper', 'Single-use carrier bag', 
    'Polypropylene bag', 'Crisp packet', 'Spread tub', 'Tupperware', 'Disposable food container', 
    'Foam food container', 'Other plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 
    'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Plastic straw', 'Paper straw', 
    'Styrofoam piece', 'Unlabeled litter', 'Cigarette', 'bag', 'metal', 'plastic', 'trash', 'trashC'
]



# Create the output folder if it doesn't exist
output_folder = 'output_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize video capture (0 for default camera, replace with CCTV stream URL if needed)
cap = cv2.VideoCapture(0)  # Replace with your CCTV stream URL

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract detection results
    boxes = results[0].boxes
    detected_classes = [model.names[int(cls)] for cls in boxes.cls]

    # Check if the frame contains any of the classes of interest and draw bounding boxes
    frame_contains_interest = False
    for i, cls in enumerate(detected_classes):
        if cls in classes_of_interest:
            frame_contains_interest = True
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            label = f"{cls} {boxes.conf[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame to the output folder if it contains any of the classes of interest
    if frame_contains_interest:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

    frame_count += 1

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()