import os
import cv2
from ultralytics import YOLO
import time
import random


def main():
    # 配置
    inference_size = 640
    th_conf = 0.25
    th_iou = 0.45
    model_path = r'D:/env/yolov8_cpu/YOLOv8-main/output/exp1/weights/best.pt'
    output_base_path = r'D:/env/yolov8_cpu/YOLOv8-main/runs/detect'

    # 生成新的文件夹名称
    date_str = time.strftime("%Y%m%d")
    index = 1
    while True:
        output_folder_name = f"{date_str}-{index:03d}"
        output_path = os.path.join(output_base_path, output_folder_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            break
        index += 1

    # 加载预训练的YOLOv8模型
    model = YOLO(model_path)

    # 为每个类别随机分配亮色
    num_classes = len(model.names)
    class_colors = {}
    for cls_id in range(num_classes):
        r = random.randint(128, 255)
        g = random.randint(128, 255)
        b = random.randint(128, 255)
        class_colors[cls_id] = (b, g, r)

    # 全局的物体计数状态
    global_class_count = {}

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_path, 'camera_result.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 检测结果保存的 txt 文件路径
    result_txt_path = os.path.join(output_path, 'detection_results.txt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, imgsz=inference_size, conf=th_conf, iou=th_iou)
        boxes = results[0].boxes

        # 本次检测到的物体集合
        current_detected_classes = set()

        # 统计本次检测到的每个类别的数量
        for box in boxes:
            cls = int(box.cls.item())
            label = model.names[cls]
            current_detected_classes.add(label)
            if label not in global_class_count:
                global_class_count[label] = 1
            elif label not in current_detected_classes:
                global_class_count[label] += 1

        # 创建一个半透明的面板
        overlay = frame.copy()
        panel_height = 30 * len(global_class_count) + 20
        cv2.rectangle(overlay, (10, 10), (220, 10 + panel_height), (255, 255, 255), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 在面板上显示类别和数量
        y = 40
        for label, count in global_class_count.items():
            # 检查 label 是否在 model.names 中
            if label in model.names.values():
                cls_id = list(model.names.values()).index(label)
                color = class_colors[cls_id]
                cv2.putText(frame, f'{label}: {count}', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += 30

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = model.names[cls]
            color = class_colors[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        out.write(frame)
        cv2.imshow('YOLOv8 Inference', frame)

        # 将检测结果写入 txt 文件
        with open(result_txt_path, 'w') as f:
            for label, count in global_class_count.items():
                f.write(f'{label}: {count}\n')

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('Real-time inference completed.')
    print(f'Detection results saved to: {result_txt_path}')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误信息：{e}")