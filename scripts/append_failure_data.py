"""
追加失败样本到 YOLO 数据集
"""

import json
import os
import shutil
from pathlib import Path

def append_failures(session_dir: str, dataset_dir: str = "./data/yolo_dataset"):
    labels_path = os.path.join(session_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"[错误] 未找到标注文件: {labels_path}")
        return

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    # YOLO 数据集路径
    dataset_path = Path(dataset_dir)
    images_train = dataset_path / "images" / "train"
    labels_train = dataset_path / "labels" / "train"
    
    # 确保文件夹存在
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    # 图像参数 (根据截图)
    # 注意：实际上应该从图片中读取尺寸，这里先沿用之前的硬编码或从图片获取
    # 为了保险，我们直接读图片宽高
    import cv2

    count = 0
    for filename, info in labels.items():
        if info.get("needs_labeling", True) or "actual_target" not in info:
            continue
            
        img_path = os.path.join(session_dir, filename)
        if not os.path.exists(img_path):
            continue
            
        # 读取图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # 目标点
        tx, ty = info["actual_target"]
        
        # 转换为 YOLO 格式 (类别 0 为平台目标)
        # 固定一个小框
        bw, bh = 80, 30
        cx = tx / w
        cy = ty / h
        nw = bw / w
        nh = bh / h
        
        # 生成唯一文件名防止冲突
        session_id = os.path.basename(session_dir)
        new_filename = f"{session_id}_{filename}"
        new_basename = new_filename.rsplit('.', 1)[0]
        
        # 复制图片
        shutil.copy(img_path, images_train / new_filename)
        
        # 写入标签
        label_content = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
        with open(labels_train / f"{new_basename}.txt", 'w') as f:
            f.write(label_content)
            
        print(f"  [增加] {new_filename}")
        count += 1

    print(f"\n[完成] 成功追加 {count} 个样本到 {images_train}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        append_failures(sys.argv[1])
    else:
        # 默认处理最新 session
        failures_dir = "./data/failures"
        if os.path.exists(failures_dir):
            sessions = sorted([d for d in os.listdir(failures_dir) if d.startswith("session_")])
            if sessions:
                latest = os.path.join(failures_dir, sessions[-1])
                print(f"正在处理最新 session: {latest}")
                append_failures(latest)
            else:
                print("未找到 session 目录")
        else:
            print("未找到 data/failures 目录")
