"""
失败样本收集器

在跳跃失败时自动保存截图和标注，用于后续训练
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple


class FailureCollector:
    """失败样本收集器"""
    
    def __init__(self, save_dir: str = "./data/failures"):
        self.save_dir = save_dir
        self.session_dir = ""
        self.sample_count = 0
        self.labels = {}
        
        self._init_session()
    
    def _init_session(self):
        """初始化会话目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.sample_count = 0
        self.labels = {}
        print(f"[收集器] 失败样本保存至: {self.session_dir}")
    
    def save_failure(self, 
                     image: np.ndarray, 
                     player_pos: Tuple[int, int],
                     detected_target: Optional[Tuple[int, int]] = None,
                     actual_target: Optional[Tuple[int, int]] = None):
        """
        保存失败样本
        
        Args:
            image: 跳跃前的截图
            player_pos: 小人位置
            detected_target: YOLO 检测到的目标位置
            actual_target: 实际正确的目标位置（如果知道）
        """
        self.sample_count += 1
        filename = f"fail_{self.sample_count:04d}.png"
        filepath = os.path.join(self.session_dir, filename)
        
        # 保存原始图像
        cv2.imwrite(filepath, image)
        
        # 保存标注信息
        self.labels[filename] = {
            "player_pos": list(player_pos),
            "detected_target": list(detected_target) if detected_target else None,
            "actual_target": list(actual_target) if actual_target else None,
            "needs_labeling": actual_target is None
        }
        
        # 保存标注文件
        labels_path = os.path.join(self.session_dir, "labels.json")
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)
        
        print(f"[收集器] 保存失败样本: {filename}")
        
        return filepath
    
    def get_stats(self) -> str:
        """获取统计信息"""
        needs_labeling = sum(1 for v in self.labels.values() if v.get("needs_labeling", True))
        return f"[收集器] 共 {self.sample_count} 个失败样本, {needs_labeling} 个待标注"


def convert_failures_to_yolo(failures_dir: str, output_dir: str = "./data/yolo_dataset"):
    """
    将失败样本转换为 YOLO 格式并合并到训练集
    
    需要先手动标注 labels.json 中的 actual_target
    """
    import shutil
    
    # 读取标注
    labels_path = os.path.join(failures_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"[错误] 未找到标注文件: {labels_path}")
        return
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # 统计
    added = 0
    skipped = 0
    
    # YOLO 参数
    box_width = 80
    box_height = 30
    
    for filename, info in labels.items():
        if info.get("actual_target") is None:
            skipped += 1
            continue
        
        img_path = os.path.join(failures_dir, filename)
        if not os.path.exists(img_path):
            continue
        
        # 读取图像获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        x, y = info["actual_target"]
        
        # 转换为 YOLO 格式
        center_x = x / w
        center_y = y / h
        norm_width = box_width / w
        norm_height = box_height / h
        
        # 边界检查
        center_x = max(norm_width/2, min(1 - norm_width/2, center_x))
        center_y = max(norm_height/2, min(1 - norm_height/2, center_y))
        
        # 复制到训练集
        base_name = filename.rsplit(".", 1)[0]
        dst_img = os.path.join(output_dir, "images", "train", f"fail_{base_name}.png")
        dst_label = os.path.join(output_dir, "labels", "train", f"fail_{base_name}.txt")
        
        shutil.copy(img_path, dst_img)
        
        with open(dst_label, 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        added += 1
    
    print(f"[转换] 添加 {added} 个样本到训练集, 跳过 {skipped} 个未标注样本")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        if len(sys.argv) < 3:
            print("用法: python failure_collector.py convert <failures_dir>")
            sys.exit(1)
        convert_failures_to_yolo(sys.argv[2])
    else:
        print("失败样本收集器")
        print("用法:")
        print("  python failure_collector.py convert <failures_dir>  # 转换失败样本到训练集")
