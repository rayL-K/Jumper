"""
屏幕截取模块

导出: ScreenCapture
"""

import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
import win32gui

from . import config

pyautogui.FAILSAFE = False


class ScreenCapture:
    """屏幕截取器：查找窗口并截取游戏画面"""

    def __init__(self):
        self.hwnd: Optional[int] = None
        self.game_region: Optional[Tuple[int, int, int, int]] = None

    def find_window(
        self, title: Optional[str] = None, silent: bool = False
    ) -> Optional[Tuple[int, int, int, int]]:
        """根据窗口标题查找窗口"""
        if title is None:
            title = config.WECHAT_WINDOW_TITLE

        hwnd = win32gui.FindWindow(None, title)
        if hwnd == 0:
            if not silent:
                print("[错误] 未找到游戏窗口")
            return None

        if not silent:
            print(f"[初始化] 发现窗口句柄: {hwnd}")
        self.hwnd = hwnd

        raw_region: tuple[int, int, int, int] = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = raw_region
        self.game_region = (x, y, x2 - x, y2 - y)
        return self.game_region

    def pop_window(self, silent: bool = False) -> None:
        """将游戏窗口置前"""
        if self.hwnd is None:
            return
        try:
            win32gui.ShowWindow(self.hwnd, 9)
            win32gui.SetForegroundWindow(self.hwnd)
            if not silent:
                print("[初始化] 窗口已置前并聚焦")
        except Exception as e:
            if not silent:
                print(f"[错误] 置前窗口失败: {e}")

    def capture(self, save: bool = False, silent: bool = False) -> Optional[np.ndarray]:
        """截取游戏画面"""
        if self.game_region is None:
            if self.find_window(silent=silent) is None:
                return None

        img = pyautogui.screenshot(region=self.game_region)
        bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if save:
            time.sleep(0.1)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"./data/images/{timestamp}.png"
            if cv2.imwrite(save_path, bgr_img):
                print(f"已保存截图到 {save_path}")
            else:
                print("保存截图失败")

        return bgr_img

if __name__ == "__main__":
    capture = ScreenCapture()
    region = capture.find_window("跳一跳")
    capture.pop_window()
    if region:
        print(f"找到窗口: {region}")
        capture.capture()
    else:
        print("未找到窗口")
