import os
import yaml
from collections import defaultdict

def count_class_instances(label_dir, class_names):
    class_counts = defaultdict(int)
    image_count = 0
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            image_count += 1
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
    return class_counts, image_count

def count_images(image_dir):
    count = 0
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            count += 1
    return count

def main():
    base_path = r'd:\env\yolov8_cpu\YOLOv8-main'
    dataset_yaml_path = os.path.join(base_path, 'data', 'aug_data', 'dataset.yaml')

    with open(dataset_yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    
    train_labels_dir = os.path.join(base_path, 'data', 'aug_data', 'train', 'labels')
    val_labels_dir = os.path.join(base_path, 'data', 'aug_data', 'valid', 'labels')
    test_images_dir = os.path.join(base_path, 'data', 'test')

    print(f"Counting instances for training set in: {train_labels_dir}")
    train_class_counts, train_image_count = count_class_instances(train_labels_dir, class_names)
    print(f"Counting instances for validation set in: {val_labels_dir}")
    val_class_counts, val_image_count = count_class_instances(val_labels_dir, class_names)
    print(f"Counting images for test set in: {test_images_dir}")
    test_image_count = count_images(test_images_dir)

    # Aggregate results
    all_class_data = defaultdict(lambda: {'train': 0, 'val': 0, 'test_images': 0})

    for class_name, count in train_class_counts.items():
        all_class_data[class_name]['train'] = count
    for class_name, count in val_class_counts.items():
        all_class_data[class_name]['val'] = count

    # Prepare data for the table
    table_data = []
    total_instances = 0
    total_train_images = train_image_count
    total_val_images = val_image_count
    total_test_images = test_image_count

    for class_name in class_names:
        train_count = all_class_data[class_name]['train']
        val_count = all_class_data[class_name]['val']
        
        # For the table, we're counting instances, not images for train/val
        # The '测试集数量' (test set quantity) refers to images, as per our assumption
        table_data.append({
            'name': class_name,
            'train': train_count,
            'val': val_count,
            'test': 0 # This will be updated later if we find a way to count instances in test
        })
        total_instances += train_count + val_count

    # Add a row for total images in each set
    table_data.append({
        'name': '总计 (图片数量)',
        'train': total_train_images,
        'val': total_val_images,
        'test': total_test_images
    })

    # Print the table header
    print("\n实验数据集样本分布类别名称 | 训练集数量 | 验证集数量 | 测试集数量 | 占比 (%)")
    print("--------------------------------------------------------------------")

    # Print data rows
    for row in table_data:
        if row['name'] == '总计 (图片数量)':
            # For the total row, calculate overall percentages based on image counts
            total_all_images = row['train'] + row['val'] + row['test']
            train_percent = (row['train'] / total_all_images * 100) if total_all_images > 0 else 0
            val_percent = (row['val'] / total_all_images * 100) if total_all_images > 0 else 0
            test_percent = (row['test'] / total_all_images * 100) if total_all_images > 0 else 0
            total_percent = train_percent + val_percent + test_percent
            print(f"{row['name']:<25} | {row['train']:<10} | {row['val']:<10} | {row['test']:<10} | {total_percent:.2f}")
        else:
            # For class rows, calculate percentage based on total instances of that class across train/val
            class_total = row['train'] + row['val']
            class_percentage = (class_total / total_instances * 100) if total_instances > 0 else 0
            print(f"{row['name']:<25} | {row['train']:<10} | {row['val']:<10} | {row['test']:<10} | {class_percentage:.2f}")

if __name__ == '__main__':
    main()
