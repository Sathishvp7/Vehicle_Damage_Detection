from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import torch
import uvicorn

app = FastAPI(title="YOLOv8 Vehicle Damage Detection API")

# Load model at startup (best.pt)
model = YOLO("best.pt")

# Load class names
with open("classes.txt") as f:
    class_names = [line.strip() for line in f if line.strip()]

@app.post("/predict")
def predict_damage(file: UploadFile = File(...)):
    # Read image
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Run inference
    results = model(image)
    detections = results[0].boxes

    response = []
    for box in detections:
        cls_id = int(box.cls[0].item())
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        conf = float(box.conf[0].item())
        xyxy = [float(x.item()) for x in box.xyxy[0]]
        response.append({
            "label": label,
            "confidence": conf,
            "box": xyxy
        })
    return JSONResponse(content={"detections": response})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
