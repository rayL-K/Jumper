"""
YOLO 模型训练脚本
"""

import os

from ultralytics import YOLO


def train_yolo(data_yaml: str, epochs: int = 50, img_size: int = 640):
    """训练 YOLO 模型"""
    model = YOLO("yolo11n.pt")
    
    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=10,  # 早停
        save=True,
        project="./data/models/yolo",
        name="jumper",
        exist_ok=True,
        verbose=True,
        # 数据增强
        augment=True,
        flipud=0.0,  # 不上下翻转 (游戏画面有方向性)
        fliplr=0.5,  # 左右翻转 (可以)
        mosaic=0.5,  # 减少马赛克增强
    )
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳模型: ./data/models/yolo/jumper/weights/best.pt")
    
    return results


def validate_model(model_path: str, data_yaml: str):
    """验证模型"""
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results


def export_model(model_path: str, format: str = "onnx"):
    """导出模型"""
    model = YOLO(model_path)
    model.export(format=format)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()
    
    print("=== YOLO 训练 ===")
    print(f"数据集: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"图像尺寸: {args.img_size}")
    
    train_yolo(args.data, args.epochs, args.img_size)
