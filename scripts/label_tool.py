"""
失败样本标注工具

通过点击图片标记正确的目标位置
"""

import os
import json
import cv2
import sys
from pathlib import Path


def label_failures(session_dir: str):
    """
    交互式标注失败样本
    
    操作说明:
    - 左键点击: 标记正确的目标位置
    - 's' 键: 保存并跳到下一张
    - 'n' 键: 跳过当前图片
    - 'q' 键: 退出并保存
    """
    labels_path = os.path.join(session_dir, "labels.json")
    
    # 加载现有标注
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    else:
        print(f"[错误] 未找到标注文件: {labels_path}")
        return
    
    # 获取所有图片
    images = [f for f in os.listdir(session_dir) if f.endswith('.png')]
    images.sort()
    
    if not images:
        print("[错误] 没有找到图片")
        return
    
    print(f"\n=== 失败样本标注工具 ===")
    print(f"目录: {session_dir}")
    print(f"图片数量: {len(images)}")
    print(f"\n操作说明:")
    print(f"  左键点击 - 标记目标位置")
    print(f"  's' - 保存并下一张")
    print(f"  'n' - 跳过")
    print(f"  'q' - 退出保存")
    print()
    
    current_target = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_target
        if event == cv2.EVENT_LBUTTONDOWN:
            # 反算原图坐标 (考虑缩放)
            scale = param['scale']
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            current_target = (orig_x, orig_y)
            print(f"  标记目标: ({orig_x}, {orig_y})")
    
    cv2.namedWindow("Label", cv2.WINDOW_NORMAL)
    
    idx = 0
    while idx < len(images):
        filename = images[idx]
        filepath = os.path.join(session_dir, filename)
        
        # 读取图片
        img = cv2.imread(filepath)
        if img is None:
            idx += 1
            continue
        
        # 计算缩放
        h, w = img.shape[:2]
        scale = min(800 / w, 600 / h, 1.0)
        display_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # 获取现有标注
        info = labels.get(filename, {})
        player_pos = info.get("player_pos")
        detected = info.get("detected_target")
        actual = info.get("actual_target")
        
        # 绘制现有标注
        vis = display_img.copy()
        
        if player_pos:
            px, py = int(player_pos[0] * scale), int(player_pos[1] * scale)
            cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(vis, "Player", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if detected:
            dx, dy = int(detected[0] * scale), int(detected[1] * scale)
            cv2.circle(vis, (dx, dy), 5, (0, 165, 255), -1)  # 橙色
            cv2.putText(vis, "Detected", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        if current_target:
            tx, ty = int(current_target[0] * scale), int(current_target[1] * scale)
            cv2.circle(vis, (tx, ty), 5, (0, 0, 255), -1)
            cv2.putText(vis, "Target", (tx + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif actual:
            tx, ty = int(actual[0] * scale), int(actual[1] * scale)
            cv2.circle(vis, (tx, ty), 5, (0, 0, 255), -1)
            cv2.putText(vis, "Target", (tx + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 显示状态
        status = "待标注" if info.get("needs_labeling", True) else "已标注"
        cv2.putText(vis, f"[{idx+1}/{len(images)}] {filename} - {status}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.setMouseCallback("Label", mouse_callback, {'scale': scale})
        cv2.imshow("Label", vis)
        
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('s'):  # 保存并下一张
            if current_target:
                labels[filename]["actual_target"] = list(current_target)
                labels[filename]["needs_labeling"] = False
                print(f"  [保存] {filename} -> {current_target}")
            current_target = None
            idx += 1
        elif key == ord('n'):  # 跳过
            current_target = None
            idx += 1
        elif key == ord('q'):  # 退出
            break
        elif key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()
    
    # 保存标注
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    labeled = sum(1 for v in labels.values() if not v.get("needs_labeling", True))
    print(f"\n[完成] 已标注 {labeled}/{len(labels)} 个样本")
    print(f"保存至: {labels_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 自动查找最新的 session 目录
        failures_dir = "./data/failures"
        if os.path.exists(failures_dir):
            sessions = sorted([d for d in os.listdir(failures_dir) if d.startswith("session_")])
            if sessions:
                latest = os.path.join(failures_dir, sessions[-1])
                print(f"自动选择最新目录: {latest}")
                label_failures(latest)
            else:
                print("未找到失败样本目录")
                print("用法: python label_tool.py <session_dir>")
        else:
            print("用法: python label_tool.py <session_dir>")
    else:
        label_failures(sys.argv[1])
