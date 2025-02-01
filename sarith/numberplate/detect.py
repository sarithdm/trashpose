import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.style.use('dark_background')

# Define paths and constants
model_path = 'yolov8n.pt'  # Pre-trained YOLOv8 model
data_yaml = "data.yaml"    # Dataset configuration file
image_size = 640          # Image size for the YOLO model
test_images_dir = "test/images"
save_dir = "model"        # Directory to save results

# Load YOLOv8 model for plate detection
model_plate_detection = YOLO(model_path)

# Train the model on the dataset
model_plate_detection.train(
    data=data_yaml,
    epochs=50,
    batch=32,
    imgsz=image_size,
    save_dir=save_dir,
    task='detect'  # Explicitly set task as detection
)

# Validate the model
model_plate_detection.val()

# Function to detect number plates
def detect_number_plates(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Run detection
    results = model_plate_detection(image)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Extract plate region
            plate_region = image[y1:y2, x1:x2]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'plate_region': plate_region
            })
    
    return detections

# Test detection on sample images
if __name__ == "__main__":
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process test images
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_path in test_images:
        try:
            detections = detect_number_plates(image_path)
            
            # Visualize results
            image = cv2.imread(image_path)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{det['confidence']:.2f}", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
            
            # Save results
            output_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image)
            print(f"Processed {image_path} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")