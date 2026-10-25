from ultralytics import YOLO
import os
import cv2
import time
from PIL import Image, ImageDraw  # 添加PIL库导入用于绘图

# 配置参数
inference_size = 640
th_conf = 0.25
th_iou = 0.35
model_path = r'D:/env/yolov8_cpu/YOLOv8-main/output/exp1/weights/best.pt'
input_path = r'D:/env/yolov8_cpu/YOLOv8-main/data/test'
output_base_path = r'D:/env/yolov8_cpu/YOLOv8-main/runs/detect'

# 自动生成带日期的输出目录
date_str = time.strftime("%Y%m%d")
index = 1
while True:
    output_folder_name = f"{date_str}-{index:03d}"
    output_path = os.path.join(output_base_path, output_folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        break
    index += 1

# 创建图像输出和标签输出子目录
image_save_dir = os.path.join(output_path, 'images')
label_save_dir = os.path.join(output_path, 'labels')
video_save_dir = os.path.join(output_path, 'videos')
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(label_save_dir, exist_ok=True)
os.makedirs(video_save_dir, exist_ok=True)

# 加载模型
try:
    model = YOLO(model_path)
    print(f"模型加载成功: {model_path}")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 图片文件扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
# 视频文件扩展名
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')


# 处理单个图片
def process_image(filename):
    try:
        img_path = os.path.join(input_path, filename)
        save_img_path = os.path.join(image_save_dir, filename)
        save_txt_path = os.path.join(label_save_dir, os.path.splitext(filename)[0] + '.txt')

        # 推理
        results = model(
            source=img_path,
            conf=th_conf,
            iou=th_iou,
            imgsz=inference_size,
            save=False,
            verbose=False
        )
        result = results[0]

        # 保存检测可视化图
        # 使用try-except处理可能的绘图错误
        try:
            img_with_boxes = result.plot()
            cv2.imwrite(save_img_path, img_with_boxes)
        except Exception as e:
            print(f"绘制图像失败 {filename}: {e}")
            # 尝试使用备用方法绘制
            img = cv2.imread(img_path)
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(save_img_path, img)

        # 保存YOLO格式的txt标注
        with open(save_txt_path, 'w') as f:
            if result.boxes is not None and result.boxes.data.numel() > 0:
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                for box in result.boxes.data.tolist():
                    cls_id, x1, y1, x2, y2, conf = int(box[5]), box[0], box[1], box[2], box[3], box[4]
                    # 转换为YOLO格式
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"图片处理完成：{filename}")
        return True
    except Exception as e:
        print(f"处理图片 {filename} 时出错: {e}")
        return False


# 处理单个视频
def process_video(filename):
    try:
        video_path = os.path.join(input_path, filename)
        save_video_path = os.path.join(video_save_dir, filename)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {filename}")
            return False

        # 获取视频的帧率和尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

        # 处理视频帧
        frame_index = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 显示进度
            frame_index += 1
            elapsed_time = time.time() - start_time
            fps_current = frame_index / elapsed_time if elapsed_time > 0 else 0
            progress = (frame_index / frame_count) * 100 if frame_count > 0 else 0
            print(
                f"\r视频处理中：{filename} - 进度: {progress:.2f}% ({frame_index}/{frame_count}) - 当前FPS: {fps_current:.2f}",
                end="")

            # 推理
            results = model(
                source=frame,
                conf=th_conf,
                iou=th_iou,
                imgsz=inference_size,
                save=False,
                verbose=False
            )
            result = results[0]

            # 绘制检测结果
            try:
                frame_with_boxes = result.plot()
            except Exception as e:
                print(f"\n绘制帧失败: {e}")
                frame_with_boxes = frame
                # 尝试备用方法绘制
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{model.names[cls_id]} {conf:.2f}"
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    2)

            # 写入视频
            out.write(frame_with_boxes)

        # 释放资源
        cap.release()
        out.release()

        print(f"\n视频处理完成：{filename}")
        return True
    except Exception as e:
        print(f"\n处理视频 {filename} 时出错: {e}")
        # 尝试释放资源
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        return False


# 批量处理图像和视频
processed_files = 0
total_files = len([f for f in os.listdir(input_path) if f.lower().endswith(image_extensions + video_extensions)])

if total_files == 0:
    print(f"未找到支持的文件在目录: {input_path}")
else:
    print(f"开始处理 {total_files} 个文件...")

    for filename in os.listdir(input_path):
        if filename.lower().endswith(image_extensions):
            success = process_image(filename)
            if success:
                processed_files += 1
        elif filename.lower().endswith(video_extensions):
            success = process_video(filename)
            if success:
                processed_files += 1
        else:
            print(f"跳过不支持的文件：{filename}")

    print(f"\n处理完成: {processed_files}/{total_files} 文件成功处理")
    print(f"结果保存在：{output_path}")
