"""
鼠标控制模块

导出: MouseController
"""

import random
import time

import pyautogui

from . import config
from .screen_capture import ScreenCapture

pyautogui.FAILSAFE = False


class MouseController:
    """鼠标控制器：执行跳跃长按操作"""

    game_region: tuple[int, int, int, int] | None
    
    def __init__(self, game_region: tuple[int, int, int, int] | None = None):
        self.game_region = game_region

    def set_game_region(self, region: tuple[int, int, int, int]) -> None:
        self.game_region = region

    def get_click_position(self) -> tuple[int, int] | None:
        """获取游戏区域内的随机点击位置"""
        if self.game_region is None:
            # 如果没设置区域，重新查找窗口并置前
            screen_capture = ScreenCapture()
            self.game_region = screen_capture.find_window(config.WECHAT_WINDOW_TITLE)
            if self.game_region is None:
                return None
            else:
                screen_capture.pop_window()
                time.sleep(config.BEFORE_CLICK_DELAY)

        x, y, width, height = self.game_region

        # 从 config 读取随机范围
        x_min_ratio, x_max_ratio = config.CLICK_RANDOM_X_RANGE
        y_min_ratio, y_max_ratio = config.CLICK_RANDOM_Y_RANGE

        rand_x = random.randint(int(x + width * x_min_ratio), int(x + width * x_max_ratio))
        rand_y = random.randint(int(y + height * y_min_ratio), int(y + height * y_max_ratio))

        return (rand_x, rand_y)
    
    def jump(self, press_time_ms: float) -> bool:
        """执行跳跃(鼠标长按)"""
        try:
            original_pos = pyautogui.position()
            result = self.get_click_position()
            if result is None:
                return False
            click_x, click_y = result
            time.sleep(config.BEFORE_CLICK_DELAY)
            press_time_sec = press_time_ms / 1000.0
            print(f"[跳跃] 位置=({click_x}, {click_y}), 按压时间={press_time_sec:.3f}s")

            pyautogui.moveTo(click_x, click_y, duration=config.MOUSE_MOVE_DURATION)
            pyautogui.mouseDown(button='left')
            time.sleep(press_time_sec)
            pyautogui.mouseUp(button='left')
            pyautogui.moveTo(original_pos[0], original_pos[1], duration=0.1)
            return True
        except Exception as e:
            print(f"[错误] 跳跃失败: {e}")
            return False
    
    def test_jump(self, press_time_ms: float = 500) -> None:
        """测试跳跃功能"""
        print(f"\n[测试] 将在3秒后执行测试跳跃（按压{press_time_ms}ms）")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        _ = self.jump(press_time_ms)
        print("[测试] 跳跃测试完成！")


if __name__ == "__main__":
    print("=" * 50)
    print("鼠标控制模块测试")
    print("=" * 50)
    controller = MouseController()
    controller.test_jump(press_time_ms=500)
