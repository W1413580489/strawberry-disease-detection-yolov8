from ultralytics import YOLO
import torch

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. 【修改这里】加载真实的 YOLO 模型
# 这里的 'yolov8n.pt' 可以换成你需要的权重文件，比如 'yolov8s.pt' 或你的自定义权重 'best.pt'
model = YOLO('yolov8n.pt').to(device)

# 3. 接下来的训练代码...
# model.train(...)
