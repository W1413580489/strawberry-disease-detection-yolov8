# 🍓 基于YOLOv8的草莓病害检测系统

基于 **YOLOv8 (Ultralytics 8.0.29 修改版)** 的草莓病害目标检测项目，支持模型训练、图片/视频/摄像头实时检测，并提供 PyQt5 图形界面。

## 检测类别（7类）

| ID | 英文名 | 中文名 |
|----|--------|--------|
| 0 | Angular_Leafspot | 角斑病 |
| 1 | Anthracnose_Fruit_Rot | 炭疽病（果实） |
| 2 | Blossom_Blight | 花疫病 |
| 3 | Gray_Mold | 灰霉病 |
| 4 | Leaf_Spot | 叶斑病 |
| 5 | Powdery_Mildew_Fruit | 白粉病（果实） |
| 6 | Powdery_Mildew_Leaf | 白粉病（叶片） |

## 目录结构

```
├── train.py                  # 训练脚本（训练完成后自动导出ONNX）
├── predict.py                # 图片/视频批量推理
├── camera.py                 # 摄像头实时检测
├── pyqt.py                   # PyQt5 图形界面（图片/视频/摄像头）
├── pt-onnx.py                # 将 .pt 权重导出为 ONNX
├── fuzhu.py / exp.py / GPU.py / runtime.py / count_*.py   # 辅助工具脚本
├── requirements.txt          # Python 依赖清单
├── ultralytics/              # 修改版框架代码（基于 Ultralytics 8.0.29）
├── yolov8s.pt                # 预训练权重（用于训练）
├── data/
│   └── aug_data/             # 数据集（YOLO格式，训练2000张/验证500张）
│       ├── dataset.yaml      # 数据集配置 ★需要修改路径★
│       ├── train/images + train/labels
│       └── valid/images + valid/labels
└── output/
    └── exp9/weights/best.pt  # 已训练好的模型权重（测试用）
```

---

## 一、下载本项目

**方式1：git 克隆（推荐）**

```bash
git clone https://github.com/W1413580489/strawberry-disease-detection-yolov8.git
cd strawberry-disease-detection-yolov8
```

**方式2：直接下载压缩包**

打开仓库网页 → 点击绿色 **Code** 按钮 → **Download ZIP** → 解压到任意目录（路径中建议不要包含中文和空格）。

---

## 二、搭建 Python 虚拟环境

### 2.1 环境要求

| 项目 | 要求 | 本项目实际测试环境 |
|------|------|-------------------|
| Python | >= 3.9（建议 3.12） | Python 3.12.7 |
| PyTorch | >= 1.7 | 2.5.1+cu121 |
| 操作系统 | Windows / Linux | Windows 10/11 |
| GPU（可选） | CUDA 12.1 | NVIDIA GPU（无GPU可改用CPU版PyTorch） |

### 2.2 创建虚拟环境

**方法A：使用 venv（Python 自带）**

```bash
python -m venv venv
# Windows 激活：
venv\Scripts\activate
# Linux/macOS 激活：
source venv/bin/activate
```

**方法B：使用 conda**

```bash
conda create -n strawberry python=3.12 -y
conda activate strawberry
```

### 2.3 安装 PyTorch（先装这个，再装其他依赖）

**有 NVIDIA 显卡（CUDA 12.1，与作者环境一致）：**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

**无显卡 / 仅用 CPU：**

```bash
pip install torch==2.5.1 torchvision==0.20.1
```

### 2.4 安装其余依赖

```bash
# 国内建议使用清华镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# GUI 需要 PyQt5（requirements.txt 未包含，必须单独安装）
pip install PyQt5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.5 关于 ultralytics 框架（重要）

> 本项目 `ultralytics/` 文件夹是基于 **Ultralytics 8.0.29 修改过的框架源码**（新增了 C2f_GhostConv 等自定义模块），**不需要也不要 `pip install ultralytics`**。
> 只要在项目根目录下运行脚本，Python 会自动优先使用项目内的这份修改版代码。

作者环境关键包版本（复现失败时可对照安装）：

```
Python 3.12.7, torch 2.5.1+cu121, torchvision 0.20.1+cu121,
numpy 1.26.4, opencv-python 4.10.0.84, PyQt5 5.15.11, pandas 3.0.1, matplotlib 3.10.8
```

---

## 三、修改数据集路径（必做）

打开 `data/aug_data/dataset.yaml`，把前两行改成 **你自己电脑上** 的实际路径：

```yaml
train: D:/你的解压路径/strawberry-disease-detection-yolov8/data/aug_data/train/images
val:   D:/你的解压路径/strawberry-disease-detection-yolov8/data/aug_data/valid/images
nc: 7
names: [Angular_Leafspot, Anthracnose_Fruit_Rot, Blossom_Blight, Gray_Mold, Leaf_Spot, Powdery_Mildew_Fruit, Powdery_Mildew_Leaf]
```

> 注意：路径分隔符用 `/`（不要用 `\`），且必须是绝对路径。

---

## 四、使用方法

### 4.1 训练模型

先编辑 `train.py` 开头的路径配置，改成自己的路径：

```python
model_path = "D:/你的路径/yolov8s.pt"                 # 预训练权重
data_path = "D:/你的路径/data/aug_data/dataset.yaml"  # 数据集配置
project_dir = "D:/你的路径/output"                    # 训练结果保存目录
```

然后运行（默认 imgsz=640, epochs=100, batch=4）：

```bash
python train.py
```

- 训练结果保存在 `output/expN/`（N 自动递增），最佳权重为 `output/expN/weights/best.pt`
- 训练完成后会自动将 best.pt 导出为 ONNX 格式
- 显存不足时：把 `batch_size` 改小（如 2）

### 4.2 图片/视频批量检测

编辑 `predict.py` 中的模型路径和输入文件夹路径后运行：

```bash
python predict.py
```

结果保存在 `runs/detect/` 目录。

### 4.3 摄像头实时检测

```bash
python camera.py
```

- 默认打开摄像头 0，按 `q` 退出
- 检测结果实时显示并录制保存为 `camera_result.mp4`
- 可在 `main()` 中修改 `model_path`、置信度阈值 `th_conf`（默认0.25）、IOU阈值 `th_iou`（默认0.45）

### 4.4 图形界面（推荐新手使用）

```bash
python pyqt.py
```

功能：
- 选择任意 `.pt` 模型文件
- 输入源支持：本地图片 / 视频 / 摄像头
- 滑块实时调整置信度阈值
- 检测结果自动保存到 `runs/detect/`

### 4.5 导出 ONNX 模型

编辑 `pt-onnx.py` 中的模型路径后运行：

```bash
python pt-onnx.py
```

导出前需安装：`pip install onnx onnxruntime onnxslim`

### 4.6 直接使用已训练好的权重测试

仓库已附带训练好的权重 `output/exp9/weights/best.pt`，可以跳过训练，直接在 `pyqt.py` / `camera.py` / `predict.py` 中把模型路径指向它进行检测验证。

---

## 五、数据集格式

采用 YOLO 检测格式，每张图片对应一个同名 `.txt` 标注文件：

```
<class_id> <x_center> <y_center> <width> <height>
```

坐标均为归一化值（0~1）。类别ID与上表 0~6 对应。

---

## 六、常见问题

<details>
<summary><b>1. 训练时报显存不足（CUDA out of memory）</b></summary>

把 `train.py` 中 `batch_size` 改为 2 或 1；仍然不足可把 `imgsz` 改为 480。
</details>

<details>
<summary><b>2. ModuleNotFoundError: No module named 'ultralytics'</b></summary>

确认你是在**项目根目录**下运行的脚本；检查项目根目录下是否存在 `ultralytics/` 文件夹（它就是框架本体，不是缺失的依赖）。
</details>

<details>
<summary><b>3. 运行 pyqt.py 报 No module named 'PyQt5'</b></summary>

`pip install PyQt5`
</details>

<details>
<summary><b>4. 摄像头打不开</b></summary>

`camera.py` 和 `pyqt.py` 默认使用摄像头编号 0，如果有多个摄像头可把 `cv2.VideoCapture(0)` 改为 `cv2.VideoCapture(1)` 等。
</details>

<details>
<summary><b>5. ONNX 导出失败 cannot import name 'ONNX_ML'</b></summary>

onnx 相关包损坏或版本不匹配，重新安装：

```bash
pip uninstall -y onnx onnxruntime onnxslim
pip install onnx onnxruntime onnxslim
```
</details>

---

## 七、致谢

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（v8.0.29）修改实现，感谢开源社区。
