
from ultralytics import YOLO

# Load a model
model = YOLO("./checkpoints/yolov8s_tune_prune.pt")  # load a pretrained model (recommended for training)
# Train the model with 2 GPUs
model.train(data="./data/UAV_tune.yaml", project='FineTune' ,epochs=15, imgsz=640, device=0, cos_lr=True, dropout=0.5, amp=False)

model.export(format="onnx", batch=1)

## train -> tune : Model summary: 225 layers, 11135987 parameters, 11135971 gradients, 28.6 GFLOPs

## tune -> prune : Model summary (fused): 168 layers, 9581520 parameters, 12883 gradients, 20.8 GFLOPs

