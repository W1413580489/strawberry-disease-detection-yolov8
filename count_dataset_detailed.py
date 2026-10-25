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

def count_labels(label_dir):
    counts = Counter()
    if not os.path.exists(label_dir):
        return counts
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        counts[class_id] += 1
    return counts

train_labels_dir = "D:/env/yolov8_cpu/YOLOv8-main/data/aug_data/train/labels"
valid_labels_dir = "D:/env/yolov8_cpu/YOLOv8-main/data/aug_data/valid/labels"

train_counts = count_labels(train_labels_dir)
valid_counts = count_labels(valid_labels_dir)

total_train = sum(train_counts.values())
total_valid = sum(valid_counts.values())

print(f"{'Class Name':<25} | {'Train':<8} | {'Val':<8} | {'Train %':<10} | {'Val %':<10}")
print("-" * 70)

for i, name in enumerate(class_names):
    t_count = train_counts[i]
    v_count = valid_counts[i]
    t_perc = (t_count / total_train * 100) if total_train > 0 else 0
    v_perc = (v_count / total_valid * 100) if total_valid > 0 else 0
    print(f"{name:<25} | {t_count:<8} | {v_count:<8} | {t_perc:>8.2f}% | {v_perc:>8.2f}%")

print("-" * 70)
print(f"{'Total Instances':<25} | {total_train:<8} | {total_valid:<8} | {'100.00%':>9} | {'100.00%':>9}")
