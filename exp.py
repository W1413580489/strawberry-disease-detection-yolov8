from ultralytics import YOLO
import os

# 定义训练好的模型路径
model_path = r"D:/env/yolov8_cpu/YOLOv8-main/output/exp3/weights/best.pt"

# 加载训练好的YOLOv8模型
model = YOLO(model_path)

# 获取模型所在的目录
model_dir = os.path.dirname(model_path)

# 构建导出的ONNX模型完整路径
onnx_path = os.path.join(model_dir, 'best.onnx')

# 导出为ONNX格式
model.export(format='onnx', name=os.path.splitext(onnx_path)[0])

# 检查导出的ONNX文件是否存在
if os.path.exists(onnx_path):
    print(f"模型成功导出为ONNX格式，导出路径为: {onnx_path}")
else:
    print("模型导出失败。")