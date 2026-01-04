"""
数据采集模块

功能：
1. 监听鼠标按压时间
2. 截取跳跃前后截图
3. 保存采集数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Any
from pynput import mouse

from screen_capture import ScreenCapture


class DataCollector:
    """跳跃数据采集器"""

    def __init__(self, session_dir: Optional[str] = None) -> None:
        """
        初始化采集器

        参数:
            session_dir: 会话目录，不指定则自动创建
        """
        self.screen = ScreenCapture()
        
        # 会话目录
        if session_dir:
            self.session_dir = session_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = f"./data/calibration/{timestamp}"
        
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 采集状态
        self._press_start: float = 0
        self._current_press_time: float = 0
        
        # 样本数据
        self.samples: List[dict[str, Any]] = []
        self.metadata_path = f"{self.session_dir}/metadata.json"

    def collect(self, num_samples: int = 5) -> None:
        """
        采集跳跃数据

        参数:
            num_samples: 采集样本数量
        """
        print("=" * 50)
        print("[数据采集] 跳跃数据采集")
        print("=" * 50)
        print(f"[信息] 会话目录: {self.session_dir}")
        print("[提示] 在游戏中正常跳跃，脚本自动记录")
        print("[提示] y=有效 n=无效 q=结束")
        print("-" * 50)

        sample_count = 0

        while sample_count < num_samples:
            print(f"\n[等待] 第 {sample_count + 1}/{num_samples} 次跳跃...")

            # 查找游戏窗口
            self.screen.hwnd = None
            self.screen.game_region = None
            if self.screen.find_window("跳一跳") is None:
                print("[错误] 找不到游戏窗口")
                continue

            self.screen.pop_window()
            time.sleep(0.3)

            # 跳跃前截图
            before_images: List[np.ndarray] = []
            for _ in range(2):
                img = self.screen.capture()
                if img is not None:
                    before_images.append(img)
                time.sleep(0.05)

            if len(before_images) == 0:
                print("[错误] 截图失败")
                continue

            # 等待跳跃
            press_time = self._wait_for_jump()
            if press_time is None:
                print("[错误] 监听失败")
                continue

            print(f"[检测] 按压时间: {press_time:.0f} ms")

            # 等待落地
            time.sleep(0.23)

            # 跳跃后连拍
            after_images: List[np.ndarray] = []
            for _ in range(5):
                img = self.screen.capture()
                if img is not None:
                    after_images.append(img)
                time.sleep(0.01)

            if len(after_images) == 0:
                print("[错误] 截图失败")
                continue

            # 确认状态
            status = input("[确认] 有效跳跃? (y/n/q): ").strip().lower()

            if status == 'q':
                print("[完成] 结束采集")
                break
            elif status != 'y':
                print("[跳过] 无效跳跃")
                continue

            # 保存截图
            before_paths = self._save_images(before_images, sample_count, "before")
            after_paths = self._save_images(after_images, sample_count, "after")

            # 记录样本
            self.samples.append({
                "before_images": before_paths,
                "after_images": after_paths,
                "press_time": press_time,
                "selected_before": None,
                "selected_after": None,
                "start_pos": None,
                "end_pos": None,
                "distance": None
            })

            sample_count += 1
            print(f"[成功] 已采集 {sample_count} 个样本")

        # 保存元数据
        self._save_metadata()
        print(f"\n[完成] 共采集 {len(self.samples)} 个样本")
        print(f"[信息] 数据目录: {self.session_dir}")

    def _wait_for_jump(self) -> Optional[float]:
        """等待一次跳跃，返回按压时间(ms)"""
        self._current_press_time = 0
        jump_completed = False

        def on_click(
            x: int, y: int, button: mouse.Button, pressed: bool
        ) -> Optional[bool]:
            nonlocal jump_completed

            if button == mouse.Button.left:
                if pressed:
                    self._press_start = time.time()
                else:
                    duration = (time.time() - self._press_start) * 1000
                    if duration > 100:  # 过滤短点击
                        self._current_press_time = duration
                        jump_completed = True
                        return False
            return None

        listener = mouse.Listener(on_click=on_click)
        listener.start()

        start_wait = time.time()
        while not jump_completed:
            if time.time() - start_wait > 30:
                listener.stop()
                return None
            time.sleep(0.1)

        listener.stop()
        return self._current_press_time

    def _save_images(
        self, images: List[np.ndarray], sample_idx: int, prefix: str
    ) -> List[str]:
        """保存图片并返回路径列表"""
        paths: List[str] = []
        for i, img in enumerate(images):
            path = f"{self.session_dir}/sample_{sample_idx}_{prefix}_{i}.png"
            cv2.imwrite(path, img)
            paths.append(path)
        return paths

    def _save_metadata(self) -> None:
        """保存元数据"""
        data = {
            "session_dir": self.session_dir,
            "created_at": datetime.now().isoformat(),
            "samples": self.samples
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="跳跃数据采集")
    parser.add_argument("-n", "--num", type=int, default=5, help="采集样本数量")
    parser.add_argument("-d", "--dir", type=str, default=None, help="会话目录")
    args = parser.parse_args()
    
    collector = DataCollector(session_dir=args.dir)
    collector.collect(args.num)
