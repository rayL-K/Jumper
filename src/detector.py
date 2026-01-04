"""
检测模块

功能：小人定位 (模板匹配) + 目标平台检测 (YOLO) + 游戏状态识别
"""

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import (
    PLAYER_TEMPLATE_PATH,
    RANKING_LIST_PATH,
    RESTART_BUTTON_PATH,
    RETURN_BUTTON_PATH,
    START_GAME_BUTTON_PATH,
)

YOLO_MODEL_PATH = "./data/models/yolo/jumper/weights/best.pt"

# 导入 YOLO
YOLO: Any = None
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # noqa: F811
    YOLO_AVAILABLE = True
except ImportError:
    pass


class Detector:
    """
    核心检测器：小人定位 + 目标平台检测 + 游戏状态识别

    检测策略:
    1. 小人定位: 模板匹配 (TM_CCOEFF_NORMED)
    2. 目标检测: YOLO 目标检测
    """

    def __init__(self) -> None:
        # 模板资源
        self.player_template = self._load_template(PLAYER_TEMPLATE_PATH, "小人")
        self.restart_template = self._load_template(RESTART_BUTTON_PATH)
        self.ranking_template = self._load_template(RANKING_LIST_PATH)
        self.return_button_template = self._load_template(RETURN_BUTTON_PATH)
        self.start_game_template = self._load_template(START_GAME_BUTTON_PATH)
        self.num_templates: Dict[int, np.ndarray] = {}
        self._load_num_templates()

        # YOLO 模型
        self.yolo_model: Any = None
        self._init_yolo()

    @staticmethod
    def _load_template(path: str, name: str = "") -> Optional[np.ndarray]:
        """加载单个模板图片"""
        if not os.path.exists(path):
            return None
        template = cv2.imread(path)
        if template is not None and name:
            h, w = template.shape[:2]
            print(f"[加载] {name}模板: {w}x{h}")
        return template

    def _init_yolo(self) -> None:
        """初始化 YOLO 模型"""
        if not YOLO_AVAILABLE or YOLO is None:
            print("[YOLO] ultralytics 未安装")
            return
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"[YOLO] 模型文件不存在: {YOLO_MODEL_PATH}")
            return
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[YOLO] 模型加载成功: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")

    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        核心检测入口

        Returns:
            (player_pos, target_pos): 小人底部中心 + 目标平台中心
        """
        player_pos = self._detect_player(image)
        if player_pos is None:
            return None, None

        target_pos = self._detect_target_yolo(image, player_pos)
        return player_pos, target_pos

    def _detect_target_yolo(
        self, image: np.ndarray, player_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        YOLO 目标检测

        策略: 过滤小人附近 + 评分排序 (置信度 - 距离偏差)
        """
        if self.yolo_model is None:
            return None

        try:
            results = self.yolo_model.predict(image, verbose=False, conf=0.3)
            if not results or not hasattr(results[0], "boxes") or not len(results[0].boxes):
                return None

            px, py = player_pos
            candidates = []

            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                conf = float(box.conf[0])

                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                # 过滤: 距离太近 或 不在小人上方
                if dist < 50 or cy > py - 30:
                    continue

                # 评分: 置信度高 + 距离适中(约200px)优先
                score = conf * 100 - abs(dist - 200) * 0.1
                candidates.append(((cx, cy), score))

            if not candidates:
                return None
            return max(candidates, key=lambda x: x[1])[0]

        except Exception as e:
            print(f"[YOLO] 推理错误: {e}")
            return None


    def _detect_player(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """模板匹配检测小人底部中心位置"""
        if self.player_template is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.player_template, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.5:
            return None
        
        h, w = template_gray.shape
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h
        return (center_x, center_y)

    def is_game_over(self, image: np.ndarray) -> bool:
        """检测是否结束"""
        if self.restart_template is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.restart_template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > 0.6

    def is_ranking_list(self, image: np.ndarray) -> bool:
        """检测是否处于排行榜界面"""
        if self.ranking_template is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.ranking_template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > 0.6

    def get_restart_button_pos(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """获取重新开始按钮位置"""
        if self.restart_template is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.restart_template, cv2.COLOR_BGR2GRAY)
        
        best_val, best_loc, best_scale = 0, None, 1.0
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            h, w = template_gray.shape
            resized = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val, best_loc, best_scale = max_val, max_loc, scale
        
        if best_val < 0.6 or best_loc is None:
            return None
        
        h, w = self.restart_template.shape[:2]
        tw, th = int(w * best_scale), int(h * best_scale)
        return (best_loc[0] + tw // 2, best_loc[1] + th // 2)

    def get_return_button_pos(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """获取返回按钮位置"""
        if self.return_button_template is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.return_button_template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.6:
            return None
        h, w = template_gray.shape
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)

    def get_start_game_button_pos(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """获取开始游戏按钮位置"""
        if self.start_game_template is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.start_game_template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.6:
            return None
        h, w = template_gray.shape
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)

    def visualize(self, image: np.ndarray, 
                  player_pos: Optional[Tuple[int, int]] = None,
                  target_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        可视化检测结果
        - 绿色圆点: 小人底部中心
        - 红色圆点: 目标平台顶面中心
        - 紫色线: 跳跃路径
        - 文字标注: 距离信息
        """
        vis = image.copy()
        
        # 绘制小人位置 (绿色)
        if player_pos:
            cv2.circle(vis, player_pos, 4, (0, 255, 0), -1)
            cv2.circle(vis, player_pos, 5, (255, 255, 255), 1)
            cv2.putText(vis, "P", (player_pos[0] + 8, player_pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制目标位置 (红色)
        if target_pos:
            cv2.circle(vis, target_pos, 4, (0, 0, 255), -1)
            cv2.circle(vis, target_pos, 5, (255, 255, 255), 1)
            cv2.putText(vis, "T", (target_pos[0] + 8, target_pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制连线和距离
        if player_pos and target_pos:
            cv2.line(vis, player_pos, target_pos, (255, 0, 255), 2)
            
            # 计算距离
            dist = int(np.sqrt((target_pos[0] - player_pos[0])**2 + 
                               (target_pos[1] - player_pos[1])**2))
            
            # 在连线中点显示距离
            mid_x = (player_pos[0] + target_pos[0]) // 2
            mid_y = (player_pos[1] + target_pos[1]) // 2
            cv2.putText(vis, f"{dist}px", (mid_x, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 显示检测方式
        method = "YOLO" if self.yolo_model else "CV"
        cv2.putText(vis, f"[{method}]", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return vis


    def _load_num_templates(self) -> None:
        """加载0-9数字模板用于分数识别"""
        num_dir = "./data/nums"
        if not os.path.exists(num_dir):
            return
        for i in range(10):
            path = os.path.join(num_dir, f"{i}.png")
            if os.path.exists(path):
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.num_templates[i] = template

    def _detect_numbers(self, roi: np.ndarray) -> int:
        """从 ROI 中识别数字"""
        if not self.num_templates:
            return -1

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        detections = []
        for num, template in self.num_templates.items():
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # 提高阈值减少误检
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                detections.append({"num": num, "x": pt[0], "y": pt[1], "conf": result[pt[1], pt[0]]})
        
        if not detections:
            return -1
        
        detections.sort(key=lambda d: d["x"])
        
        # 去重
        filtered = []
        last_x = -100
        for d in detections:
            if d["x"] - last_x > 15:
                filtered.append(d)
                last_x = d["x"]
        
        score_str = "".join(str(d["num"]) for d in filtered)
        try:
            return int(score_str)
        except:
            return -1

    def get_score(self, image: np.ndarray) -> int:
        """识别实时得分 (左上角 ROI)"""
        h, w = image.shape[:2]
        roi = image[int(h * 0.05):int(h * 0.30), int(w * 0.02):int(w * 0.50)]
        return self._detect_numbers(roi)

    def get_final_score(self, image: np.ndarray) -> int:
        """识别最终得分 (游戏结束界面中上部分)"""
        h, w = image.shape[:2]
        roi = image[int(h * 0.10):int(h * 0.40), int(w * 0.10):int(w * 0.90)]
        return self._detect_numbers(roi)
