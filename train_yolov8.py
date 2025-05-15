import os
from ultralytics import YOLO

# Path to your dataset YAML file
data_yaml = 'data_custom.yaml'

# Choose a YOLOv8 model variant (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
# You can download pretrained weights or use 'yolov8n.pt' for a lightweight model
model = YOLO('yolov8n.pt')  # Downloaded automatically if not present

# Train the model
results = model.train(
    data=data_yaml,          # Path to dataset YAML
    epochs=50,              # Number of training epochs
    imgsz=640,              # Image size
    batch=16,               # Batch size
    project='yolo8_runs',   # Output directory
    name='custom_exp_aug',  # New experiment name for augmentation
    device=0,               # Use GPU 0; set to 'cpu' for CPU
    # Augmentation settings
    hsv_h=0.1,   # HSV-Hue augmentation
    hsv_s=0.7,   # HSV-Saturation augmentation
    hsv_v=0.4,   # HSV-Value augmentation
    degrees=10.0, # Image rotation
    translate=0.1, # Image translation
    scale=0.5,    # Image scaling
    shear=2.0,    # Shear angle
    perspective=0.0, # Perspective
    flipud=0.5,   # Vertical flip
    fliplr=0.5,   # Horizontal flip
    mosaic=1.0,   # Mosaic augmentation probability
    mixup=0.2,    # MixUp augmentation probability
)

# Evaluate the model on validation set
eval_results = model.val()

# Save the trained model
model.export(format='torchscript')  # Exports to TorchScript format
print('Training and export complete.')
