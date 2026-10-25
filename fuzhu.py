import os
import cv2
from ultralytics import YOLO
import time
import random
import numpy as np

def main():
    # 配置
    inference_size = 640
    th_conf = 0.25
    th_iou = 0.35
    model_path = r'D:/env/yolov8_cpu/YOLOv8-main/output/exp1/weights/best.pt'
    input_path = r'D:/env/yolov8_cpu/YOLOv8-main/data/test'
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

    # 批量推理
    file_extensions = ('.jpg', '.png', '.jpeg', '.mp4', '.avi')
    files = [f for f in os.listdir(input_path) if f.endswith(file_extensions)]

    for file in files:
        file_path = os.path.join(input_path, file)
        if file.endswith(('.jpg', '.png', '.jpeg')):
            # 处理图片
            image = cv2.imread(file_path)
            height, width = image.shape[:2]
            region_height = height // 4  # 计算每个区域的高度

            results = model.predict(file_path, imgsz=inference_size, conf=th_conf, iou=th_iou)

            boxes = results[0].boxes
            region_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 初始化区域统计

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = box.conf.item()
                cls = int(box.cls.item())
                label = model.names[cls]
                color = class_colors[cls]

                # 计算中心点坐标
                cy = (y1 + y2) // 2
                # 确定所属区域
                region = cy // region_height
                if region >= 4:
                    region = 3  # 处理下边界情况
                region_counts[region] += 1  # 更新统计

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 绘制区域划分线
            for i in range(1, 4):
                y = i * region_height
                cv2.line(image, (0, y), (width, y), (0, 0, 255), 2)

            # 绘制区域统计信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0  # 增大字体大小
            thickness = 3     # 增大字体厚度
            for i in range(4):
                text = f'R{i + 1}: {region_counts[i]}'
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = 10  # 固定左侧距离
                y = i * region_height + 40  # 固定上方距离，避开区域分界线

                # 创建半透明背景矩形
                overlay = image.copy()
                cv2.rectangle(overlay, (x - 5, y - text_height - 5),
                              (x + text_width + 5, y + 5),
                              (0, 0, 0), -1)  # 黑色填充矩形
                alpha = 0.5  # 设置透明度
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # 绘制大红色文字
                cv2.putText(image, text, (x, y), font, font_scale,
                            (0, 0, 255), thickness)  # 大红色

            cv2.imwrite(os.path.join(output_path, f'{file}_result.jpg'), image)
            print(f'Processed {file} and saved results to {output_path}')
        elif file.endswith(('.mp4', '.avi')):
            # 处理视频
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = os.path.join(output_path, f'{file}_result.mp4')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                region_height = h // 4

                results = model.predict(frame, imgsz=inference_size, conf=th_conf, iou=th_iou)
                boxes = results[0].boxes
                region_counts = {0: 0, 1: 0, 2: 0, 3: 0}

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = model.names[cls]
                    color = class_colors[cls]

                    cy = (y1 + y2) // 2
                    region = cy // region_height
                    if region >= 4:
                        region = 3
                    region_counts[region] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # 绘制区域划分线
                for i in range(1, 4):
                    y = i * region_height
                    cv2.line(frame, (0, y), (w, y), (0, 0, 255), 2)

                # 绘制区域统计信息
                for i in range(4):
                    text = f'R{i + 1}: {region_counts[i]}'
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                    x = 10
                    y = i * region_height + 20

                    # 创建半透明背景矩形
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x - 5, y - text_height - 5),
                                  (x + text_width + 5, y + 5),
                                  (0, 0, 0), -1)
                    alpha = 0.5
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # 绘制大红色文字
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 255), 3)

                out.write(frame)

            cap.release()
            out.release()
            print(f'Processed {file} and saved results to {output_path}')

    print('Batch inference completed.')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误信息：{e}")