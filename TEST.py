
import os
import torch
from ultralytics import YOLO
import torchprofile

# 加载 .pt 文件
weight_path = "./runs/detect/train51/weights/last.pt"
weight_path = "./checkpoints/yolov8s_train.pt"
weight_path = "./checkpoints/model_pruned.pt"
weight_path = "./checkpoints/yolov8s_tune.pt"
# weight_path = './FineTune/train5/weights/last.pt'

# 重新实例化模型
model = YOLO(weight_path).model  # 加载基础模型

# 定义输入张量的形状（例如，对于YOLOv8，通常是 [1, 3, 640, 640]）
dummy_input = torch.randn(1, 3, 640, 640).to(next(model.parameters()).device)

# 使用 torchprofile 计算FLOPs
flops = torchprofile.profile_macs(model, dummy_input)
print(f"Model FLOPs: {flops / 1e9} GFLOPs")  # 将FLOPs转换为GFLOPs (10^9)

# 验证图片路径
PATH = './datasets/UAV_tune/images/val/'

# List all image files in the validation folder
image_files = [os.path.join(PATH, img) for img in os.listdir(PATH) if img.endswith(('.jpg', '.jpeg', '.png'))]

# 重新实例化模型
model = YOLO(weight_path)  # 使用YOLO接口加载模型

# Run batched inference on the list of images
results = model(image_files)  # 返回结果列表

directory_path = './TEST_result/'
os.makedirs(directory_path, exist_ok=True)
# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # 绑定框输出
    masks = result.masks  # 分割掩码输出
    keypoints = result.keypoints  # 关键点输出
    probs = result.probs  # 分类概率输出
    obb = result.obb  # 定向框输出
    result.show()  # 显示结果
    result.save(filename=f"{directory_path}result_{i}.jpg")  # 保存结果








# import os
# from ultralytics import YOLO
# import torch

# from ptflops import get_model_complexity_info


# # 加载 .pt 文件
# weight_path = "./runs/detect/train51/weights/last.pt"
# weight_path = "./checkpoints/yolov8s_train.pt"
# weight_path = "./checkpoints/model_pruned.pt"
# weight_path = './FineTune/train5/weights/last.pt'

# # 重新实例化模型
# model = YOLO(weight_path)  # 加载基础模型

# PATH = './datasets/UAV_tune/images/val/'

# # List all image files in the validation folder
# image_files = [os.path.join(PATH, img) for img in os.listdir(PATH) if img.endswith(('.jpg', '.jpeg', '.png'))]

# # Run batched inference on the list of images
# results = model(image_files)  # return a list of Results objects

# # Process results list
# for i, result in enumerate(results):
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename=f"result_{i}.jpg")  # save to disk
