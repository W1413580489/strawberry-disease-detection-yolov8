import os
from collections import Counter

class_names = [
    "Angular_Leafspot",
    "Anthracnose_Fruit_Rot",
    "Blossom_Blight",
    "Gray_Mold",
    "Leaf_Spot",
    "Powdery_Mildew_Fruit",
    "Powdery_Mildew_Leaf"
]

def count_images_per_class(label_dir):
    counts = Counter()
    if not os.path.exists(label_dir):
        return counts, 0
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    total_images = len(label_files)
    
    for label_file in label_files:
        unique_classes_in_image = set()
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    unique_classes_in_image.add(class_id)
        
        for class_id in unique_classes_in_image:
            counts[class_id] += 1
            
    return counts, total_images

train_labels_dir = "D:/env/yolov8_cpu/YOLOv8-main/data/aug_data/train/labels"
valid_labels_dir = "D:/env/yolov8_cpu/YOLOv8-main/data/aug_data/valid/labels"

train_counts, total_train_imgs = count_images_per_class(train_labels_dir)
valid_counts, total_valid_imgs = count_images_per_class(valid_labels_dir)

print(f"{'Class Name':<25} | {'Train Imgs':<10} | {'Val Imgs':<10} | {'Train %':<10} | {'Val %':<10}")
print("-" * 75)

for i, name in enumerate(class_names):
    t_count = train_counts[i]
    v_count = valid_counts[i]
    # 占比计算：该类别的图片数量 / 该数据集总图片数量
    t_perc = (t_count / total_train_imgs * 100) if total_train_imgs > 0 else 0
    v_perc = (v_count / total_valid_imgs * 100) if total_valid_imgs > 0 else 0
    print(f"{name:<25} | {t_count:<10} | {v_count:<10} | {t_perc:>8.2f}% | {v_perc:>8.2f}%")

print("-" * 75)
print(f"{'Total Unique Images':<25} | {total_train_imgs:<10} | {total_valid_imgs:<10} | {'100.00%':>9} | {'100.00%':>9}")
print("\n注：由于一张图片可能包含多个类别的目标，各类别图片数量之和可能大于总图片数量。")
