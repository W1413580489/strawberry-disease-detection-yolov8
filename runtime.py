import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
import yaml

class YOLOv8:
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres, yaml_path):
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = self.get_classes_from_yaml(yaml_path)
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.initialize_session(self.onnx_model)

    def get_classes_from_yaml(self, yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        except FileNotFoundError:
            print(f"Error: YAML file {yaml_path} not found.")
            return []
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return []

    def initialize_session(self, onnx_model):
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_model, session_options=session_options, providers=providers)

    def preprocess(self):
        self.img = cv2.imread(self.input_image)
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        x_factor = self.img_width / 640
        y_factor = self.img_height / 640

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][:4]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)
        return input_image

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def main(self):
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        img_data = self.preprocess()
        outputs = self.session.run(None, {model_inputs[0].name: img_data})
        return self.postprocess(self.img, outputs)

if __name__ == "__main__":
    onnx_model_name = r"D:/env/yolov8_cpu/YOLOv8-main/output/exp3/weights/best.onnx"
    test_image_folder = r"D:env\yolov8_cpu\YOLOv8-main\data\test"
    output_folder = r"D:\env\yolov8_cpu\YOLOv8-main\data\output"
    yaml_path = r"D:\env\yolov8_cpu\YOLOv8-main\data\aug_data\dataset.yaml"

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历测试图片文件夹中的所有图片
    for filename in os.listdir(test_image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_image_folder, filename)
            detection = YOLOv8(onnx_model_name, img_path, 0.3, 0.5, yaml_path)
            output_image = detection.main()

            # 构建输出图片的完整路径
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, output_image)
            print(f"Processed {filename} and saved to {output_image_path}")