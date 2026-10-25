from ultralytics import YOLO

# 指定.pt模型的路径
model_path = r'D:\env\yolov8_cpu\YOLOv8-main\output\exp9\weights\best.pt'

# 加载模型
model = YOLO(model_path)

# 导出模型为 ONNX 格式，导出路径默认为模型同目录下，以 best.onnx 命名
model.export(format='onnx', dynamic=True)
print(f"ONNX模型导出成功！")