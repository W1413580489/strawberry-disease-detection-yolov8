import os
from ultralytics import YOLO
import torch
import re


def get_latest_exp_folder(project_dir):
    """获取最新的exp目录（按编号排序）"""
    folders = [os.path.join(project_dir, d) for d in os.listdir(project_dir) if
               os.path.isdir(os.path.join(project_dir, d))]
    if not folders:
        return None
    # 按最后修改时间排序，返回最新的
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder
    if not exp_folders:
        return None
    exp_folders.sort(key=lambda x: int(x[3:]) if x[3:].isdigit() else -1)
    return os.path.join(project_dir, exp_folders[-1])

def train():
    # 配置参数
    # ========== 修改点 1：指定改进后的 YAML 配置文件 ==========
    model_cfg = "D:/env/yolov8_cpu/YOLOv8-main/ultralytics/models/v8/YOLOv8_LSKA_P2_GhostConv.yaml" # 模型结构文件（含 LSKA）
    pretrained_weight = "D:/env/yolov8_cpu/YOLOv8-main/yolov8s.pt"  # 预训练权重（用于迁移学习）
    # ========================================================

    train_args = {
        "data": "D:/env/yolov8_cpu/YOLOv8-main/data/aug_data/dataset.yaml",
        "device": '0' if torch.cuda.is_available() else 'cpu', 
        "project": "D:/env/yolov8_cpu/YOLOv8-main/output",
        "name": "LSKA_P2_GhostConv_Scratch",
        "imgsz": 640,
        "epochs": 150,
        "batch": 4,
        "pretrained": True,
        "optimizer": 'SGD', 
    }

    print(f"开始训练，配置文件：{model_cfg}，预训练权重：{pretrained_weight}")

    # ========== 修改点 2：先根据 YAML 构建模型，再加载预训练权重 ==========
    # 1. 根据 YAML 创建网络结构（包含 LSKA）
    model = YOLO(model_cfg)
    # 2. 加载预训练权重（strict=False 允许跳过新增的 LSKA 层）
    #model = model.load(pretrained_weight)
    # =================================================================

    # 开始训练
    results = model.train(**train_args)

    # 获取最新的exp目录
    project_dir = train_args["project"]
    latest_exp = get_latest_exp_folder(project_dir)
    if not latest_exp:
        print("未找到训练结果目录，请检查训练是否成功")
        return

    best_model_path = os.path.join(latest_exp, "weights", "best.pt")
    onnx_export_path = os.path.join(latest_exp, "weights", "best.onnx")

    if not os.path.exists(best_model_path):
        print(f"最佳模型文件不存在：{best_model_path}")
        return

    # 导出 ONNX
    try:
        print(f"正在从 {best_model_path} 导出 ONNX 模型...")
        exported_model = YOLO(best_model_path)
        exported_model.export(format="onnx", dynamic=True)
        print(f"ONNX 模型已导出至：{onnx_export_path}")
    except Exception as e:
        print(f"导出失败，错误信息：{str(e)}")
        return

    if os.path.exists(onnx_export_path):
        print(f"训练及导出完成，结果目录：{latest_exp}")
    else:
        print("ONNX 文件未生成，请检查导出日志")

if __name__ == "__main__":
    train()