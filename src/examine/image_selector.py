"""
图片挑选与标记模块

功能：
1. 为样本选择最佳跳跃前后图片
2. 使用模板匹配自动标记小人位置
3. 手动标记备选
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import numpy as np
from typing import Optional, Tuple, List, Any

from calculator import JumpCalculator


class ImageSelector:
    """图片挑选与标记器"""

    def __init__(self, session_dir: str) -> None:
        """
        初始化

        参数:
            session_dir: 会话目录
        """
        self.session_dir = session_dir
        self.metadata_path = f"{session_dir}/metadata.json"
        self.template_path = "./data/player_template.png"
        self.calculator = JumpCalculator()
        self.samples: List[dict[str, Any]] = []

    def load_metadata(self) -> bool:
        """加载元数据"""
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.samples = data.get("samples", [])
            print(f"[信息] 已加载 {len(self.samples)} 个样本")
            return True
        except FileNotFoundError:
            print(f"[错误] 找不到元数据: {self.metadata_path}")
            return False

    def save_metadata(self) -> None:
        """保存元数据"""
        # 读取现有数据
        data: dict[str, Any]
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"session_dir": self.session_dir}
        
        data["samples"] = self.samples
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


    def select_images(self) -> None:
        """
        图片挑选阶段：为每个样本选择最佳跳跃前后图片
        """
        print("=" * 50)
        print("[图片挑选] 为样本选择最佳图片对")
        print("=" * 50)
        print("[提示] A/D=切换左图 J/K=切换右图 Enter=确认 ESC=跳过")
        print("-" * 50)

        for i, sample in enumerate(self.samples):
            print(f"\n[样本 {i+1}/{len(self.samples)}] 按压: {sample['press_time']:.0f}ms")

            result = self._select_pair(
                sample["before_images"],
                sample["after_images"],
                f"样本{i+1}"
            )

            if result is None:
                print("[跳过] 未选择")
                sample["selected_before"] = None
                sample["selected_after"] = None
            else:
                sample["selected_before"] = result[0]
                sample["selected_after"] = result[1]
                print("[成功] 已选择图片对")

        self.save_metadata()
        valid = sum(1 for s in self.samples if s.get("selected_before") and s.get("selected_after"))
        print(f"\n[完成] 有效样本: {valid}/{len(self.samples)}")

    def _select_pair(
        self, before_paths: List[str], after_paths: List[str], title: str
    ) -> Optional[Tuple[str, str]]:
        """并列显示跳跃前后图片，返回选择的路径对"""
        valid_before = [p for p in before_paths if os.path.exists(p)]
        valid_after = [p for p in after_paths if os.path.exists(p)]

        if not valid_before or not valid_after:
            return None

        before_idx, after_idx = 0, 0
        window = f"{title} - A/D=left J/K=right Enter=confirm ESC=skip"

        while True:
            before_img = cv2.imread(valid_before[before_idx])
            after_img = cv2.imread(valid_after[after_idx])

            if before_img is None:
                valid_before.pop(before_idx)
                if not valid_before:
                    cv2.destroyAllWindows()
                    return None
                before_idx = before_idx % len(valid_before)
                continue

            if after_img is None:
                valid_after.pop(after_idx)
                if not valid_after:
                    cv2.destroyAllWindows()
                    return None
                after_idx = after_idx % len(valid_after)
                continue

            # 调整大小
            h1, h2 = before_img.shape[0], after_img.shape[0]
            target_h = min(h1, h2, 600)
            s1, s2 = target_h / h1, target_h / h2
            before_resized = cv2.resize(before_img, None, fx=s1, fy=s1)
            after_resized = cv2.resize(after_img, None, fx=s2, fy=s2)

            # 标签
            cv2.putText(before_resized, f"BEFORE [{before_idx+1}/{len(valid_before)}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(after_resized, f"AFTER [{after_idx+1}/{len(valid_after)}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            combined = cv2.hconcat([before_resized, after_resized])
            cv2.imshow(window, combined)
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key in (13, 32):  # Enter/Space
                cv2.destroyAllWindows()
                return (valid_before[before_idx], valid_after[after_idx])
            elif key in (ord('a'), ord('A')):
                before_idx = (before_idx - 1) % len(valid_before)
            elif key in (ord('d'), ord('D')):
                before_idx = (before_idx + 1) % len(valid_before)
            elif key in (ord('j'), ord('J')):
                after_idx = (after_idx - 1) % len(valid_after)
            elif key in (ord('k'), ord('K')):
                after_idx = (after_idx + 1) % len(valid_after)

    def mark_with_template(self) -> None:
        """使用模板匹配自动标记小人位置"""
        print("=" * 50)
        print("[模板标记] 自动标记小人位置")
        print("=" * 50)

        if not os.path.exists(self.template_path):
            print(f"[错误] 找不到模板: {self.template_path}")
            return

        template = cv2.imread(self.template_path)
        if template is None:
            print("[错误] 模板加载失败")
            return

        th, tw = template.shape[:2]
        print(f"[信息] 模板尺寸: {tw}x{th}")

        for i, sample in enumerate(self.samples):
            if not sample.get("selected_before") or not sample.get("selected_after"):
                continue

            print(f"\n[样本 {i+1}]")

            before_img = cv2.imread(sample["selected_before"])
            after_img = cv2.imread(sample["selected_after"])

            if before_img is None or after_img is None:
                continue

            # 匹配
            start_pos = self._match_template(before_img, template, f"Sample {i+1} BEFORE")
            if start_pos is None:
                start_pos = self._get_click(before_img, "点击小人底部")
                if start_pos is None:
                    continue

            end_pos = self._match_template(after_img, template, f"Sample {i+1} AFTER")
            if end_pos is None:
                end_pos = self._get_click(after_img, "点击落点位置")
                if end_pos is None:
                    continue

            distance = self.calculator.calculate_distance(start_pos, end_pos)

            sample["start_pos"] = start_pos
            sample["end_pos"] = end_pos
            sample["distance"] = distance

            print(f"[成功] 距离: {distance:.1f}px 时间: {sample['press_time']:.0f}ms")

        cv2.destroyAllWindows()
        self.save_metadata()
        print("\n[完成] 标记完成")

    def _match_template(
        self, image: np.ndarray, template: np.ndarray, title: str
    ) -> Optional[Tuple[int, int]]:
        """模板匹配，返回小人底部中心坐标"""
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        print(f"  匹配度: {max_val:.3f}")

        if max_val < 0.6:
            return None

        th, tw = template.shape[:2]
        center_x = max_loc[0] + tw // 2
        bottom_y = max_loc[1] + th - 5

        # 可视化
        display = image.copy()
        cv2.rectangle(display, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 0, 255), 2)
        cv2.circle(display, (center_x, bottom_y), 5, (0, 255, 0), -1)

        h = display.shape[0]
        scale = min(600 / h, 1.0)
        resized = cv2.resize(display, None, fx=scale, fy=scale)
        cv2.imshow(title, resized)
        cv2.waitKey(500)

        return (center_x, bottom_y)

    def _get_click(self, image: np.ndarray, title: str) -> Optional[Tuple[int, int]]:
        """手动点击获取位置"""
        clicked: Optional[Tuple[int, int]] = None

        def callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked = (x, y)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, callback)

        display = image.copy()
        cv2.putText(display, "Click to mark, ESC to skip", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(title, display)

        while True:
            key = cv2.waitKey(100)
            if key == 27:
                cv2.destroyAllWindows()
                return None
            if clicked is not None:
                cv2.circle(display, clicked, 5, (0, 0, 255), -1)
                cv2.imshow(title, display)
                cv2.waitKey(300)
                break

        cv2.destroyAllWindows()
        return clicked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="图片挑选与标记")
    parser.add_argument("-d", "--dir", type=str, required=True, help="会话目录")
    parser.add_argument("--mark", action="store_true", help="执行模板标记")
    args = parser.parse_args()

    selector = ImageSelector(args.dir)
    if not selector.load_metadata():
        exit(1)

    if args.mark:
        selector.mark_with_template()
    else:
        selector.select_images()
