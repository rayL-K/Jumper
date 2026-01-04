"""
配置文件

集中管理所有硬编码参数，便于调整和维护。
"""

from typing import Final, Tuple

# ==============================================
# 游戏窗口配置
# ==============================================
WECHAT_WINDOW_TITLE: Final[str] = "跳一跳"


# ==============================================
# 版本信息
# ==============================================
VERSION: Final[str] = "3.0.0"


# ==============================================
# 跳跃计算参数
# ==============================================
# 按压系数（从校准得到: examine/coefficient_fitter.py）
PRESS_COEFFICIENT: Final[float] = 3.1361
PRESS_LOSS: Final[float] = 26.25
# 按压时间限制（毫秒），避免异常值
MIN_PRESS_TIME: Final[float] = 200.0
MAX_PRESS_TIME: Final[float] = 1000.0


# ==============================================
# 跳跃方向参数 (仅用于可视化/参考)
# ==============================================
JUMP_ANGLE_DEG: Final[int] = 60        # 跳跃方向角度（与垂直方向夹角）
JUMP_LINE_LENGTH: Final[int] = 350     # 辅助线长度


# ==============================================
# 模板路径
# ==============================================
PLAYER_TEMPLATE_PATH: Final[str] = "./data/player_template.png"
RESTART_BUTTON_PATH: Final[str] = "./data/restart_button.png"
RANKING_LIST_PATH: Final[str] = "./data/ranking_list.png"
RETURN_BUTTON_PATH: Final[str] = "./data/return_button.png"
START_GAME_BUTTON_PATH: Final[str] = "./data/start_game_button.png"


# ==============================================
# 控制器配置
# ==============================================
# 随机点击区域比例
CLICK_RANDOM_X_RANGE: Final[Tuple[float, float]] = (0.3, 0.7)
CLICK_RANDOM_Y_RANGE: Final[Tuple[float, float]] = (0.4, 0.6)

# 延时参数（秒）
BEFORE_CLICK_DELAY: Final[float] = 0.1
MOUSE_MOVE_DURATION: Final[float] = 0.1


# ==============================================
# 游戏时间参数（秒）
# ==============================================
LANDING_WAIT_TIME: Final[float] = 1.5 # 落地等待时间
RESTART_WAIT_TIME: Final[float] = 1.0  # 重新开始等待时间


# ==============================================
# 调试与异常处理
# ==============================================
SHOW_DEBUG_WINDOW: Final[bool] = True    # 是否实时显示调试画面
FAILURE_SAVE_DIR: Final[str] = "./data/failures"  # 识别失败截图保存目录

# ==============================================
# 数据采集
# ==============================================
SAVE_DATA_ENABLED: Final[bool] = True     # 是否在运行过程中采集数据
DATA_SAVE_DIR: Final[str] = "./data/calibration" # 数据采集根目录


# (深度学习配置已移至 detector.py YOLO 管理)
