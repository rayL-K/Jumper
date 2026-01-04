"""
YOLO 数据准备脚本

将 labels.json 中的目标点坐标转换为 YOLO 检测格式
"""

import json
import os
import shutil
import random
from pathlib import Path


def convert_to_yolo_format(labels_path: str, calibration_dir: str, output_dir: str):
    """
    将 labels.json 转换为 YOLO 目标检测格式
    
    YOLO 格式: class_id center_x center_y width height (归一化)
    
    由于我们只有目标点坐标，我们将创建一个小的边界框围绕该点
    """
    # 创建输出目录
    output_path = Path(output_dir)
    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"
    
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 读取标签
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # 统计
    total = len(labels)
    valid = 0
    skipped = 0
    
    # 图像尺寸 (根据之前检查的结果)
    img_width = 464
    img_height = 847
    
    # 边界框大小 (固定大小，以目标点为中心)
    # 平台顶面大约 60-100 像素宽，20-40 像素高
    box_width = 80
    box_height = 30
    
    all_items = list(labels.items())
    random.shuffle(all_items)
    
    # 80% 训练, 20% 验证
    split_idx = int(len(all_items) * 0.8)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    
    def process_items(items, images_dir, labels_dir, prefix=""):
        nonlocal valid, skipped
        for rel_path, coords in items:
            # 修复路径分隔符
            rel_path = rel_path.replace("\\", "/")
            img_path = Path(calibration_dir) / rel_path
            
            if not img_path.exists():
                skipped += 1
                continue
            
            # 目标坐标
            x, y = coords
            
            # 转换为 YOLO 格式 (归一化的中心坐标和尺寸)
            center_x = x / img_width
            center_y = y / img_height
            norm_width = box_width / img_width
            norm_height = box_height / img_height
            
            # 确保在有效范围内
            center_x = max(norm_width/2, min(1 - norm_width/2, center_x))
            center_y = max(norm_height/2, min(1 - norm_height/2, center_y))
            
            # 生成文件名
            safe_name = rel_path.replace("/", "_").replace("\\", "_")
            base_name = safe_name.rsplit(".", 1)[0]
            
            # 复制图像
            dst_img = images_dir / f"{base_name}.png"
            shutil.copy(img_path, dst_img)
            
            # 写入标签文件
            # 类别 0 = target (目标平台中心)
            label_content = f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n"
            dst_label = labels_dir / f"{base_name}.txt"
            with open(dst_label, 'w') as f:
                f.write(label_content)
            
            valid += 1
    
    print(f"处理训练集 ({len(train_items)} 张)...")
    process_items(train_items, images_train, labels_train)
    
    print(f"处理验证集 ({len(val_items)} 张)...")
    process_items(val_items, images_val, labels_val)
    
    # 创建 data.yaml 配置文件
    yaml_content = f"""# Jumper YOLO Dataset
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: target
"""
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n=== 转换完成 ===")
    print(f"总计: {total} 条记录")
    print(f"有效: {valid} 张图像")
    print(f"跳过: {skipped} 张 (文件不存在)")
    print(f"训练集: {len(train_items)} 张")
    print(f"验证集: {len(val_items)} 张")
    print(f"配置文件: {yaml_path}")
    
    return str(yaml_path)


if __name__ == "__main__":
    labels_path = "./data/calibration/labels.json"
    calibration_dir = "./data/calibration"
    output_dir = "./data/yolo_dataset"
    
    convert_to_yolo_format(labels_path, calibration_dir, output_dir)
