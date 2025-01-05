from ultralytics import YOLO
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

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

# Connect to MongoDB using the URI from the environment variable
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['your_database_name']
collection = db['your_collection_name']

# Insert keypoints into MongoDB
if keypoints is not None:
    keypoints_list = keypoints.tolist()  # Convert numpy array to list
    document = {'keypoints': keypoints_list}
    collection.insert_one(document)
    print("Keypoints inserted into MongoDB")
else:
    print("No keypoints to insert")