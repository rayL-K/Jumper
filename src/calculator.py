"""
距离与时间计算模块

导出: JumpCalculator
"""

import math
from typing import Tuple

from . import config

Point = Tuple[float, float]


class JumpCalculator:
    """跳跃计算器：根据距离计算按压时间"""

    def __init__(self, press_coefficient: float | None = None) -> None:
        self.press_coefficient = press_coefficient or config.PRESS_COEFFICIENT

    def calculate_distance(self, player: Point, target: Point) -> float:
        """计算两点间的欧几里得距离"""
        x1, y1 = player
        x2, y2 = target
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
    def calculate_press_time(self, distance: float) -> float:
        """根据距离计算需要的按压时间 (毫秒)"""
        corrected_distance = max(0, distance - config.PRESS_LOSS)
        press_time = corrected_distance * self.press_coefficient
        return max(config.MIN_PRESS_TIME, min(config.MAX_PRESS_TIME, press_time))

    def adjust_coefficient(self, new_coefficient: float) -> None:
        """调整系数（用于参数调优）"""
        old = self.press_coefficient
        self.press_coefficient = new_coefficient
        print(f"[调参] 系数从 {old} 调整为 {new_coefficient}")

if __name__ == "__main__":
    calc = JumpCalculator(3.4543)
    test_cases = [
        ((100, 500), (300, 400)),
        ((200, 600), (400, 500)),
        ((150, 700), (500, 500)),
    ]
    print(f"\n当前系数: {calc.press_coefficient}")
    for player, target in test_cases:
        distance = calc.calculate_distance(player, target)
        press_time = calc.calculate_press_time(distance)
        print(f"{player} → {target}: 距离={distance:.1f}px, 按压={press_time:.0f}ms")
