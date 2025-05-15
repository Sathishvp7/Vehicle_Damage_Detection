# YOLOv8 Custom Object Detection Project

This repository contains all code, configuration, and data structure for training a custom object detection model using YOLOv8 (Ultralytics) with PyTorch.

## Directory Structure

```
yolo_8_detection/
├── analyze_label_counts.py        # Script to analyze class label distribution
├── best.pt                       # Best trained YOLOv8 weights (with augmentation)
├── best_without_augumentation.pt  # Best trained weights (without augmentation)
├── classes.txt                   # List of class names (one per line)
├── data_custom.yaml              # Dataset configuration for YOLOv8
├── requirements.txt              # Python dependencies
├── train_yolov8.py               # Main training script
├── train/                        # Training images and labels
│   ├── images/
│   └── labels/
├── val/                          # Validation images and labels
│   ├── images/
│   └── labels/
├── runs/                         # YOLOv8 experiment outputs (predictions, logs)
│   └── detect/
└── README.md                     # Project documentation (this file)
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_yolov8.py
```

- The script will train YOLOv8 on your custom dataset with augmentation enabled.
- The best model weights will be saved as `best.pt`.

### 3. Analyze Class Distribution
```bash
python analyze_label_counts.py
```

- Prints the number of instances per class in train/val splits.

### 4. Inference / Prediction
You can use the best weights for inference with YOLOv8's API or CLI:
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('path/to/image_or_folder')
```

### 5. Project Notes
- All scripts are modular and ready for further extension (e.g., for preprocessing, advanced analysis, or deployment).
- For assignment/reporting, see the comments in `train_yolov8.py` and the summary in this README.


## REST API for Model Inference

A FastAPI-based REST API (`app.py`) is included to serve the trained YOLOv8 model for vehicle damage detection.

### How to Use the API

1. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the API server:**
   ```bash
   uvicorn app:app --reload
   ```
   The API will be available at [http://localhost:8000/docs](http://localhost:8000/docs) (interactive Swagger UI).

### API Endpoint

- **POST /predict**
  - Upload an image file (form-data, key: `file`)
  - Returns: Detected damages with labels, confidence scores, and bounding box coordinates

#### Example Request (Python)
```python
import requests

url = 'http://localhost:8000/predict'
image_path = 'test.jpg'
with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
    print(response.json())
```

#### Example Response
```json
{
  "detections": [
    {
      "label": "dent",
      "confidence": 0.92,
      "box": [x1, y1, x2, y2]
    },
    ...
  ]
}
```

## About the Dataset
- Classes: scratch, dent, broken_light
- Data is organized in YOLO format: images and label text files for each split.

## Contact
For questions or improvements, please contact sathishvp7@gmail.com

## Credit
Kaggle dataset for vechicle damage detection - 
https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset?resource=download

Ultralytics YOLOv8 - 
https://docs.ultralytics.com/

PyTorch
https://pytorch.org/

Annotation tool
labelImg(Python pip)
