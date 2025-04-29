from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == "__main__":
    # Load a model
    model = YOLO(model="yolo11n.pt")  # build a new model from YAML

    # Train the model
    results = model.train(data="sar_aircraft.yaml", epochs=10, imgsz=1024, batch=16)
