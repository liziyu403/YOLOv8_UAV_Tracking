from ultralytics import YOLO
import torchprofile
import torch
import os

# Load the exported ONNX model
model = YOLO("./checkpoints/best.onnx", task='detect') 

# Path to the validation images
PATH = './datasets/UAV_tune/images/val/'

# List all image files in the validation folder
image_files = [os.path.join(PATH, img) for img in os.listdir(PATH) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image individually
for i, img_path in enumerate(image_files):
    # Run inference on the individual image
    results = model(img_path)  # Returns a result object

    # Process results
    for result in results:
        boxes = result.boxes  # Bounding box output
        masks = result.masks  # Segmentation mask output
        keypoints = result.keypoints  # Keypoints output
        probs = result.probs  # Classification probabilities output
        obb = result.obb  # Oriented bounding box output
        result.show()  # Display result
        result.save(filename=f"./ONNX_detection_results/result_{i}.jpg")  # Save result
