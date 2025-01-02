from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load pretrained model for pose estimation
model = YOLO('yolov8n-pose.pt')  # You can use a different model version like yolov8l-pose.pt

# Predict poses from an image
results = model('throw_image.jpg')  # replace '66742559.jpg' with the path to your image
#print(results)


# Access the first result and use the plot() method to get the image with keypoints
img_with_keypoints = results[0].plot()  # plot() returns the annotated image

# Display the image using matplotlib
plt.imshow(img_with_keypoints)
plt.axis('off')  # Hide the axes for a cleaner look
plt.show()