import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QStatusBar,
                             QFileDialog, QMessageBox, QProgressBar, QTextEdit,
                             QComboBox, QSlider, QGroupBox, QGridLayout, QSpinBox,
                             QSplitter, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 修改为相对路径 - 结果保存在当前目录下的runs/detect文件夹中
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_SAVE_DIR = os.path.join(current_dir, "runs", "detect")
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)  # 确保目录存在


class DetectionThread(QThread):
    """检测线程，避免UI卡顿"""
    update_signal = pyqtSignal(int, str, np.ndarray, list)
    finish_signal = pyqtSignal(str, int)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, model_path, input_path, conf_thres=0.25,
                 input_type='image', save_frames=False, save_video=True, frame_interval=5):
        super().__init__()
        self.model_path = model_path
        self.input_path = input_path  # 摄像头模式下为整数（如0）
        self.current_conf_thres = conf_thres  # 存储当前实际使用的阈值
        self.iou_thres = 0.45  # 固定IOU阈值
        self.input_type = input_type  # 'image', 'video', 'camera'
        self.save_frames = save_frames  # 是否保存单帧图片
        self.save_video = save_video  # 是否保存视频
        self.frame_interval = frame_interval  # 摄像头保存帧间隔
        self.images = []
        self.results = []
        self.current_index = 0
        self.stopped = False
        self.save_dir = self._get_save_dir()
        self.video_writer = None  # 视频写入器
        self.file_counter = {}  # 用于跟踪同名文件的计数
        self.cap = None  # 摄像头捕获对象
        self.annotator = None  # YOLOv8的注释器

    def _get_save_dir(self):
        """根据日期生成唯一输出目录，同一天的结果保存在同一个文件夹中"""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(ROOT_SAVE_DIR, date_str)

    def _get_exp_counter(self):
        """获取当前输入文件的实验计数器"""
        if self.input_type == 'camera':
            base_name = 'camera'
        else:
            base_name = os.path.basename(str(self.input_path))
            base_name = os.path.splitext(base_name)[0] if '.' in base_name else base_name

        # 检查保存目录中已存在的实验次数
        save_dir = self._get_save_dir()
        if not os.path.exists(save_dir):
            return 1

        existing_files = os.listdir(save_dir)
        max_exp = 0
        for f in existing_files:
            if f.startswith(base_name):
                # 查找文件名中的exp数字
                parts = f.split('exp')
                if len(parts) > 1:
                    num_part = parts[1].split('_')[0]
                    if num_part.isdigit():
                        max_exp = max(max_exp, int(num_part))
        return max_exp + 1

    def _get_output_filename(self, base_name, ext, is_video=False):
        """生成带exp序号的输出文件名"""
        if self.input_type == 'camera':
            base = 'camera'
        else:
            base = os.path.splitext(base_name)[0] if '.' in base_name else base_name

        if is_video:
            return f"{base}_exp{self.exp_counter}_detected{ext}"
        else:
            return f"{base}_exp{self.exp_counter}_detected{ext}"

    def set_conf_thres(self, conf_thres):
        self.current_conf_thres = conf_thres
        print(f"检测线程阈值更新为: {self.current_conf_thres}")

    def run(self):
        try:
            # 加载YOLOv8模型
            self.model = YOLO(self.model_path)
            self.names = self.model.names
            self.colors = colors

            if self.input_type == 'camera':
                self._process_camera()
            else:
                self._process_file()

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self._release_resources()

    def _process_file(self):
        """处理图片和视频文件"""
        os.makedirs(self.save_dir, exist_ok=True)

        # 安全处理文件名
        base_name = os.path.basename(str(self.input_path))
        base_name_no_ext = os.path.splitext(base_name)[0] if '.' in base_name else base_name
        self.exp_counter = self._get_exp_counter()

        # 初始化视频写入器
        if self.save_video and self.input_type != 'image':
            video_output_path = os.path.join(self.save_dir,
                                             self._get_output_filename(base_name_no_ext, ".mp4", is_video=True))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = None  # 将在处理第一帧时初始化

        # 使用OpenCV读取输入
        if self.input_type == 'image':
            img = cv2.imread(self.input_path)
            if img is None:
                raise ValueError(f"无法读取图片: {self.input_path}")

            # 执行检测
            results = self.model.predict(img, conf=self.current_conf_thres, iou=self.iou_thres)

            # 使用YOLOv8的Annotator绘制结果
            annotated_img = img.copy()
            self.annotator = Annotator(annotated_img, line_width=2, font_size=0.5)

            detections = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = f"{self.names[cls]} {conf:.2f}"
                    detections.append((label, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))))

                    # 使用YOLOv8的Annotator绘制检测框
                    self.annotator.box_label(xyxy, label, color=self.colors(cls, True))

            annotated_img = self.annotator.result()

            # 保存结果
            if self.save_frames:
                save_path = os.path.join(self.save_dir,
                                         self._get_output_filename(base_name_no_ext, ".jpg"))
                cv2.imwrite(save_path, annotated_img)
                self.images.append(os.path.basename(save_path))
            else:
                self.images.append(f"图片")

            self.results.append((annotated_img, detections))
            self.current_index = len(self.results) - 1
            self.update_signal.emit(1, self.images[-1], annotated_img, detections)

        else:  # 视频处理
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {self.input_path}")

            frame_count = 0
            while not self.stopped and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 执行检测
                results = self.model.predict(frame, conf=self.current_conf_thres, iou=self.iou_thres)

                # 使用YOLOv8的Annotator绘制结果
                annotated_frame = frame.copy()
                self.annotator = Annotator(annotated_frame, line_width=2, font_size=0.5)

                detections = []
                for result in results:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        label = f"{self.names[cls]} {conf:.2f}"
                        detections.append((label, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))))

                        # 使用YOLOv8的Annotator绘制检测框
                        self.annotator.box_label(xyxy, label, color=self.colors(cls, True))

                annotated_frame = self.annotator.result()

                # 初始化视频写入器
                if self.save_video and self.video_writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    h, w = annotated_frame.shape[:2]
                    video_output_path = os.path.join(self.save_dir,
                                                     self._get_output_filename(base_name_no_ext, ".mp4", is_video=True))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))

                # 写入视频帧
                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(annotated_frame)

                # 保存单帧图片
                if self.save_frames and frame_count % self.frame_interval == 0:
                    save_path = os.path.join(self.save_dir,
                                             f"{base_name_no_ext}_exp{self.exp_counter}_frame_{frame_count}.jpg")
                    cv2.imwrite(save_path, annotated_frame)
                    self.images.append(os.path.basename(save_path))
                else:
                    self.images.append(f"视频帧 {frame_count}")

                self.results.append((annotated_frame, detections))
                self.current_index = len(self.results) - 1
                self.update_signal.emit(frame_count, self.images[-1], annotated_frame, detections)

                # 更新进度
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.progress_signal.emit(int(frame_count / total_frames * 100))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

        self.finish_signal.emit(self.save_dir, len(self.results))

    def _process_camera(self):
        """处理摄像头输入"""
        os.makedirs(self.save_dir, exist_ok=True)
        self.exp_counter = self._get_exp_counter()

        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头: {self.input_path}")

        frame_count = 0
        fps = 30.0  # 默认帧率
        h, w = None, None

        # 生成唯一的视频文件名
        video_output_path = os.path.join(self.save_dir,
                                         self._get_output_filename('camera', '.mp4', is_video=True))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while not self.stopped and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.error_signal.emit("无法从摄像头读取帧")
                break

            frame_count += 1

            if h is None or w is None:
                h, w = frame.shape[:2]
                if self.save_video:
                    self.video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))

            # 执行检测
            results = self.model.predict(frame, conf=self.current_conf_thres, iou=self.iou_thres)

            # 使用YOLOv8的Annotator绘制结果
            annotated_frame = frame.copy()
            self.annotator = Annotator(annotated_frame, line_width=2, font_size=0.5)

            detections = []
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = f"{self.names[cls]} {conf:.2f}"
                    detections.append((label, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))))

                    # 使用YOLOv8的Annotator绘制检测框
                    self.annotator.box_label(xyxy, label, color=self.colors(cls, True))

            annotated_frame = self.annotator.result()

            # 写入视频
            if self.save_video and self.video_writer is not None:
                self.video_writer.write(annotated_frame)

            # 保存单帧图片
            if self.save_frames and frame_count % self.frame_interval == 0:
                save_path = os.path.join(self.save_dir,
                                         f"camera_exp{self.exp_counter}_frame_{frame_count}.jpg")
                cv2.imwrite(save_path, annotated_frame)
                self.images.append(os.path.basename(save_path))
            else:
                self.images.append(f"摄像头帧 {frame_count}")

            self.results.append((annotated_frame, detections))
            self.current_index = len(self.results) - 1
            self.update_signal.emit(frame_count, self.images[-1], annotated_frame, detections)

            # 控制帧率
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC或Q键退出
                break

        self.finish_signal.emit(self.save_dir, frame_count)

    def _release_resources(self):
        """释放所有资源"""
        # 释放视频写入器
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # 释放摄像头
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def stop(self):
        self.stopped = True
        self._release_resources()


class YOLOv8GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8目标检测可视化工具")
        self.setGeometry(100, 100, 1200, 800)

        self.current_index = 0
        self.images = []
        self.results = []
        self.model = None
        self.detection_thread = None
        self.is_detecting = False
        self.conf_threshold = 0.25  # 默认置信度阈值
        self.input_type = 'image'
        self.save_frames = False  # 默认不保存单帧图片
        self.save_video = True  # 默认保存视频
        self.open_dir_btn = None  # 用于跟踪目录按钮

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        control_layout = QHBoxLayout()

        # 模型选择
        model_group = QGroupBox("模型配置")
        model_layout = QHBoxLayout(model_group)
        self.model_path_label = QLabel("模型路径: 未选择")
        self.model_path_label.setMinimumWidth(200)
        self.select_model_btn = QPushButton("选择模型")
        self.select_model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.select_model_btn)
        model_group.setLayout(model_layout)

        # 输入源选择
        input_group = QGroupBox("输入源")
        input_layout = QHBoxLayout(input_group)
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItems(["图片", "视频", "摄像头"])
        self.input_type_combo.currentIndexChanged.connect(self.on_input_type_changed)
        self.input_path_label = QLabel("路径: 未选择")
        self.input_path_label.setMinimumWidth(200)
        self.select_input_btn = QPushButton("选择")
        self.select_input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(QLabel("输入类型:"))
        input_layout.addWidget(self.input_type_combo)
        input_layout.addWidget(self.input_path_label)
        input_layout.addWidget(self.select_input_btn)
        input_group.setLayout(input_layout)

        # 阈值设置 - 仅保留置信度阈值
        threshold_group = QGroupBox("检测阈值")
        threshold_layout = QHBoxLayout(threshold_group)  # 将布局直接应用于QGroupBox

        # 置信度阈值控制
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)  # 范围设为0-100
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setMinimumWidth(300)  # 增加滑块的最小宽度
        self.conf_slider.valueChanged.connect(self.on_conf_threshold_changed)
        conf_layout.addWidget(self.conf_slider)

        self.conf_value_label = QLabel(f"{self.conf_threshold * 100:.0f}%")
        self.conf_value_label.setFixedWidth(60)
        self.conf_value_label.setAlignment(Qt.AlignCenter)
        conf_layout.addWidget(self.conf_value_label)

        # 将置信度控制添加到阈值布局
        threshold_layout.addLayout(conf_layout)

        # 创建一个容器布局用于加载设置按钮
        btn_layout = QHBoxLayout()
        self.load_settings_btn = QPushButton("加载设置")
        self.load_settings_btn.setMinimumWidth(100)
        self.load_settings_btn.clicked.connect(self.load_settings)
        btn_layout.addWidget(self.load_settings_btn)
        btn_layout.addStretch()  # 添加伸缩以使按钮居右

        threshold_layout.addLayout(btn_layout)

        threshold_group.setLayout(threshold_layout)  # 设置布局

        # 保存选项
        save_group = QGroupBox("保存选项")
        save_layout = QHBoxLayout(save_group)
        self.save_frames_checkbox = QCheckBox("保存单帧图片")
        self.save_frames_checkbox.setChecked(self.save_frames)
        self.save_frames_checkbox.stateChanged.connect(self.on_save_options_changed)

        self.save_video_checkbox = QCheckBox("保存视频文件")
        self.save_video_checkbox.setChecked(self.save_video)
        self.save_video_checkbox.stateChanged.connect(self.on_save_options_changed)

        save_layout.addWidget(self.save_frames_checkbox)
        save_layout.addWidget(self.save_video_checkbox)
        save_group.setLayout(save_layout)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)

        control_layout.addWidget(model_group)
        control_layout.addWidget(input_group)
        control_layout.addWidget(threshold_group)
        control_layout.addWidget(save_group)
        control_layout.addWidget(self.start_btn)

        self.image_label = QLabel("请选择模型和输入源并开始检测")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(100)
        self.info_text.setFont(QFont("SimHei", 10))

        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)

        self.image_counter = QLabel("0/0")
        self.image_counter.setAlignment(Qt.AlignCenter)

        self.save_btn = QPushButton("保存当前图片")
        self.save_btn.clicked.connect(self.save_current_image)
        self.save_btn.setEnabled(False)

        nav_layout.addStretch(1)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.image_counter)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.save_btn)
        nav_layout.addStretch(1)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.info_text)
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.progress_bar)

        self.setCentralWidget(main_widget)

        # 根据输入类型调整保存选项的可见性和状态
        self.on_input_type_changed(self.input_type_combo.currentIndex())

    def on_save_options_changed(self):
        self.save_frames = self.save_frames_checkbox.isChecked()
        self.save_video = self.save_video_checkbox.isChecked()

        # 确保至少选择一个保存选项
        if not self.save_frames and not self.save_video:
            sender = self.sender()
            if sender == self.save_frames_checkbox:
                self.save_video_checkbox.setChecked(True)
                self.save_video = True
            else:
                self.save_frames_checkbox.setChecked(True)
                self.save_frames = True

    def on_input_type_changed(self, index):
        self.input_type = ['image', 'video', 'camera'][index]

        # 根据输入类型调整保存选项
        if self.input_type == 'image':
            # 图片模式：必须保存单帧
            self.save_frames_checkbox.setEnabled(True)
            self.save_frames_checkbox.setChecked(True)
            self.save_video_checkbox.setEnabled(False)
            self.save_video_checkbox.setChecked(False)
            self.save_frames = True
            self.save_video = False
        elif self.input_type == 'video':
            # 视频模式：默认保存视频，可选择保存帧
            self.save_frames_checkbox.setEnabled(True)
            self.save_video_checkbox.setEnabled(True)
            self.save_video_checkbox.setChecked(True)
            self.save_frames = self.save_frames_checkbox.isChecked()
            self.save_video = True
        elif self.input_type == 'camera':
            # 摄像头模式：默认保存视频，可选择保存关键帧
            self.save_frames_checkbox.setEnabled(True)
            self.save_video_checkbox.setEnabled(True)
            self.save_video_checkbox.setChecked(True)
            self.save_frames = self.save_frames_checkbox.isChecked()
            self.save_video = True

        if self.input_type == 'camera':
            self.input_path = 0  # 摄像头索引为整数
            self.input_path_label.setText("输入源: 摄像头")
            self.check_ready_to_start()
        else:
            self.input_path_label.setText(f"路径: 未选择")
            if hasattr(self, 'input_path'):
                delattr(self, 'input_path')
            self.start_btn.setEnabled(False)

    def on_conf_threshold_changed(self, value):
        """处理置信度阈值变化 - 更新并立即应用"""
        # 允许0值
        new_conf = value / 100.0
        self.conf_threshold = new_conf
        self.conf_value_label.setText(f"{value}%")
        print(f"置信度阈值更新为: {self.conf_threshold}")

        # 如果正在检测，实时更新
        if self.is_detecting and self.detection_thread:
            self.detection_thread.set_conf_thres(new_conf)
            self.statusBar.showMessage(f"已实时更新置信度阈值: {self.conf_threshold:.2f}")

    def load_settings(self):
        """加载设置按钮的回调函数"""
        # 现在滑块调整已立即应用阈值，此按钮主要用于显示状态
        self.statusBar.showMessage(f"当前置信度阈值: {self.conf_threshold:.2f}")

    def check_ready_to_start(self):
        """检查是否可以开始检测"""
        if hasattr(self, 'model_path') and (hasattr(self, 'input_path') or self.input_type == 'camera'):
            self.start_btn.setEnabled(True)

    def select_model(self):
        """选择YOLOv8模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLOv8模型", "", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(f"模型路径: {os.path.basename(file_path)}")
            self.check_ready_to_start()

    def select_input(self):
        """选择输入文件（图片或视频）"""
        if self.input_type == 'camera':
            # 摄像头模式不需要选择文件
            return

        if self.input_type == 'image':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
            )
            if file_path:
                self.input_path = file_path
                self.input_path_label.setText(f"图片路径: {os.path.basename(file_path)}")
        elif self.input_type == 'video':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                self.input_path = file_path
                self.input_path_label.setText(f"视频路径: {os.path.basename(file_path)}")

        self.check_ready_to_start()

    def start_detection(self):
        """开始目标检测"""
        if self.is_detecting:
            self.stop_detection()
            return

        self.start_btn.setText("停止检测")
        self.is_detecting = True
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.image_counter.setText("0/0")
        self.images = []
        self.results = []
        self.current_index = 0

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.detection_thread = DetectionThread(
            self.model_path,
            self.input_path,
            conf_thres=self.conf_threshold,
            input_type=self.input_type,
            save_frames=self.save_frames,
            save_video=self.save_video
        )
        self.detection_thread.update_signal.connect(self.update_detection_result)
        self.detection_thread.finish_signal.connect(self.on_detection_finished)
        self.detection_thread.error_signal.connect(self.on_detection_error)
        self.detection_thread.progress_signal.connect(self.progress_bar.setValue)
        self.detection_thread.start()

        self.statusBar.showMessage(
            f"正在加载模型并开始{['图片', '视频', '摄像头'][['image', 'video', 'camera'].index(self.input_type)]}检测..."
        )

    def stop_detection(self):
        """停止目标检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.statusBar.showMessage("检测已停止")
            self.is_detecting = False
            self.start_btn.setText("开始检测")
            self.progress_bar.setVisible(False)
            if self.results:
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

    def update_detection_result(self, progress, img_path, img, detections):
        """更新检测结果显示"""
        self.images.append(img_path)
        self.results.append((img, detections))

        if len(self.results) == 1 or self.input_type in ['video', 'camera']:
            self.current_index = len(self.results) - 1
            self.update_image_display()

        if self.input_type == 'image':
            self.statusBar.showMessage(f"正在处理: {img_path} ({progress}/{len(self.images)})")
        elif self.input_type == 'video':
            self.statusBar.showMessage(f"正在处理视频帧: {progress}")
        elif self.input_type == 'camera':
            self.statusBar.showMessage(f"正在实时检测摄像头画面: {progress}")

    def update_image_display(self):
        """更新图像显示"""
        if not self.results:
            return

        img, detections = self.results[self.current_index]

        h, w, c = img.shape
        q_img = QImage(img.data, w, h, w * c, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        info_text = f"{'图片' if self.input_type == 'image' else '帧'}: {self.images[self.current_index]}\n"
        info_text += f"检测到目标: {len(detections)}个\n\n"
        for i, (label, box) in enumerate(detections):
            info_text += f"{i + 1}. {label}, 位置: {box}\n"

        self.info_text.setText(info_text)

        if self.input_type == 'image':
            self.prev_btn.setEnabled(self.current_index > 0)
            self.next_btn.setEnabled(self.current_index < len(self.results) - 1)
            self.image_counter.setText(f"{self.current_index + 1}/{len(self.results)}")
            self.save_btn.setEnabled(True)
        elif self.input_type in ['video', 'camera']:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.image_counter.setText(f"当前帧")
            self.save_btn.setEnabled(True)

    def prev_image(self):
        """显示上一张图片"""
        if self.current_index > 0 and self.input_type == 'image':
            self.current_index -= 1
            self.update_image_display()

    def next_image(self):
        """显示下一张图片"""
        if self.current_index < len(self.results) - 1 and self.input_type == 'image':
            self.current_index += 1
            self.update_image_display()

    def save_current_image(self):
        """保存当前显示的图片"""
        if not self.results:
            return

        img, _ = self.results[self.current_index]
        default_name = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        if self.input_type == 'image':
            default_name = f"detected_{os.path.basename(self.images[self.current_index])}"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_name, "图像文件 (*.jpg *.png *.bmp)"
        )

        if file_path:
            cv2.imwrite(file_path, img)
            QMessageBox.information(self, "保存成功", f"图片已保存至: {file_path}")

    def on_detection_finished(self, output_path, count):
        """处理检测完成事件"""
        self.progress_bar.setVisible(False)
        self.is_detecting = False
        self.start_btn.setText("开始检测")

        # 移除旧的目录按钮（如果存在）
        if self.open_dir_btn:
            self.statusBar.removeWidget(self.open_dir_btn)
            self.open_dir_btn.deleteLater()
            self.open_dir_btn = None

        # 添加新的目录按钮
        self.open_dir_btn = QPushButton("打开保存目录")
        self.open_dir_btn.clicked.connect(lambda: os.startfile(output_path))
        self.statusBar.addPermanentWidget(self.open_dir_btn)

        if self.input_type == 'image':
            self.statusBar.showMessage(f"图片检测完成! 共处理{count}张图片，结果保存在: {output_path}")
        elif self.input_type == 'video':
            self.statusBar.showMessage(f"视频检测完成! 结果保存在: {output_path}")
        elif self.input_type == 'camera':
            self.statusBar.showMessage(f"摄像头检测完成! 结果保存在: {output_path}")

        if self.results:
            self.prev_btn.setEnabled(self.input_type == 'image')
            self.next_btn.setEnabled(self.input_type == 'image')
            self.save_btn.setEnabled(True)
            self.image_counter.setText(
                f"{self.current_index + 1}/{len(self.results)}" if self.input_type == 'image' else "当前帧")

    def on_detection_error(self, error_msg):
        """处理检测错误事件"""
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage(f"检测出错: {error_msg}")
        self.is_detecting = False
        self.start_btn.setText("开始检测")
        QMessageBox.critical(self, "错误", f"检测过程中出错:\n{error_msg}")

    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        if self.results:
            self.update_image_display()
        super().resizeEvent(event)

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        if self.is_detecting:
            self.stop_detection()

        # 释放摄像头资源
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()

        event.accept()


if __name__ == "__main__":
    os.environ["QT_FONT_DPI"] = "96"
    app = QApplication(sys.argv)
    app.setFont(QFont("SimHei"))  # 设置中文字体

    window = YOLOv8GUI()
    window.show()

    sys.exit(app.exec_())