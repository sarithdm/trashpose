from ultralytics import YOLO
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Ensure MONGO_URI is loaded
mongo_uri = os.getenv('MONGO_URI')
if not mongo_uri:
    raise ValueError("MONGO_URI environment variable is not set")

# Connect to MongoDB using the URI from the environment variable
try:
    client = MongoClient(mongo_uri)
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection error: {e}")
    raise

# Load the pretrained YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Predict poses from an image
results = model('throw_image.jpg')  # replace 'throw_image.jpg' with the path to your image

# Extract keypoints from the first result
keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    vector1 = point1 - point2
    # ...existing code...

db = client['Litter']  # Use the database name from your MONGO_URI
collection = db['detections']  # Replace with your collection name

# Create the document to insert
document = {
    "timestamp": datetime.now(),
    "location": {
        "camera_id": "camera_123",
        "coordinates": {"lat": 12.9715987, "lon": 77.594566},
        "area": "Main Street"
    },
    "detected_objects": [
        {
            "class": "person",
            "confidence": 0.98,
            "bounding_box": [100, 150, 200, 250]
        }
    ],
    "pose_estimation": {
        "keypoints": keypoints.tolist() if keypoints is not None else [],
        "action_detected": "throwing garbage"
    },
    "image_details": {
        "image_path": "path/to/throw_image.jpg",
        "image_size": {"width": 1920, "height": 1080}
    },
    "alert_status": {
        "is_alert_triggered": True,
        "alert_message": "Garbage throwing detected",
        "reported_to": ["admin@example.com"]
    },
    "vehicle_details": {
        "number_plate": "KA01AB1234",
        "confidence": 0.95,
        "vehicle_class": "car",
        "bounding_box": [300, 400, 500, 600]
    }
}

# Insert the document into MongoDB
collection.insert_one(document)
print("Document inserted into MongoDB")