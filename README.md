# Jumper - 微信 跳一跳 自动化脚本

<div align="center">

**基于 YOLO 目标检测的微信跳一跳游戏自动化脚本**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)

</div>

---

## 项目概述

本项目实现了微信"跳一跳"小程序的自动化操作。通过计算机视觉技术检测小人位置和目标平台位置，计算最优按压时长，自动执行精准跳跃。

### 业务场景

跳一跳的核心机制：按住屏幕蓄力，松开后跳跃。按压时间越长，跳跃距离越远。**核心挑战**是根据目标距离精确计算按压时长。

### 技术栈

| 分类         | 技术方案                 |
| ------------ | ------------------------ |
| **开发语言** | Python 3.11+             |
| **目标检测** | YOLOv11 (Ultralytics)    |
| **图像处理** | OpenCV (模板匹配)        |
| **屏幕截取** | PyAutoGUI + Win32 API    |
| **输入控制** | PyAutoGUI (鼠标按压)     |
| **深度学习** | PyTorch 2.6+ (CUDA 12.4) |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       主循环 (main.py)                           │
├─────────────────────────────────────────────────────────────────┤
│  ScreenCapture  →  Detector  →  Calculator  →  Controller       │
│    (截屏模块)       (检测模块)    (计算模块)     (控制模块)         │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  YOLO 检测   │
                    │  目标平台    │
                    └──────────────┘
```

---

## 核心技术实现

### 1. 目标检测策略 (`src/detector.py`)

采用 **YOLO 目标检测** 识别目标平台：

```python
def detect(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """核心检测入口"""
    player_pos = self._detect_player(image)  # 模板匹配定位小人
    if player_pos is None:
        return None, None

    target_pos = self._detect_target_yolo(image, player_pos)
    return player_pos, target_pos
```

**设计要点**: 小人使用模板匹配（外观固定，速度快），目标平台使用 YOLO（形状多变，需要学习）。

### 2. YOLO 候选目标评分算法

```python
def _detect_target_yolo(self, image, player_pos):
    # ... YOLO 推理 ...
    for box in results[0].boxes:
        cx, cy = 计算边界框中心(box)
        dist = 计算到小人的距离(cx, cy, player_pos)
        
        # 过滤: 排除小人附近 + 必须在小人上方
        if dist < 50 or cy > py - 30:
            continue

        # 评分: 置信度高 + 距离适中(约200px)优先
        score = confidence * 100 - abs(dist - 200) * 0.1
        candidates.append(((cx, cy), score))

    return max(candidates, key=lambda x: x[1])[0]
```

**面试要点**: 评分函数平衡了检测置信度和几何约束（偏好 200px 左右的跳跃距离，这是典型的跳跃范围）。

### 3. 按压时长计算 (`src/calculator.py`)

```python
def calculate_press_time(self, distance: float) -> float:
    """将像素距离转换为按压时长 (毫秒)"""
    corrected_distance = max(0, distance - PRESS_LOSS)
    press_time = corrected_distance * self.press_coefficient
    return clamp(press_time, MIN_PRESS_TIME, MAX_PRESS_TIME)
```

`press_coefficient` 通过数据采集和线性回归拟合得到。距离与按压时间呈近似线性关系。

### 4. 小人定位 - 模板匹配

```python
def _detect_player(self, image):
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val < 0.5:  # 置信度阈值
        return None
    
    # 返回小人底部中心坐标
    return (max_loc[0] + width // 2, max_loc[1] + height)
```


---

## 项目主文件结构

```
jumper/
├── main.py                     # 主循环入口
├── pyproject.toml              # 项目配置与依赖
│
├── src/                        # 核心模块
│   ├── __init__.py             # 模块导出
│   ├── detector.py             # 目标检测器 (YOLO + 模板匹配)
│   ├── calculator.py           # 距离→按压时间计算
│   ├── controller.py           # 鼠标输入控制
│   ├── screen_capture.py       # 窗口截图
│   ├── config.py               # 配置常量
│   ├── failure_collector.py    # 失败样本收集
│   └── examine/                # 调参工具
│       ├── coefficient_fitter.py   # 系数拟合
│       ├── data_collector.py       # 数据采集
│       └── image_selector.py       # 图片标注
│
├── scripts/                    # 脚本工具
│   ├── train_yolo.py           # YOLO 模型训练
│   ├── label_tool.py           # 失败样本标注工具
│   ├── append_failure_data.py  # 追加失败样本到数据集
│   └── prepare_yolo_data.py    # 数据集准备
│
└── data/                       # 数据目录 (gitignore)
    ├── models/yolo/            # 训练好的 YOLO 权重
    ├── player_template.png     # 小人模板图片
    ├── nums/                   # 数字模板 (分数识别)
    └── failures/               # 失败样本
```

---

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+
- 微信桌面版，运行跳一跳游戏

### 安装

```bash
# 克隆项目
git clone https://github.com/rayL_K/jumper.git
cd jumper

# 安装依赖 (推荐使用 uv)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 正式运行

```bash
# 启动脚本
python main.py

# 按 ESC 键停止
```

---

## 训练步骤

1. **收集失败样本** - 游戏过程中自动收集
2. **标注样本**:
   ```bash
   python scripts/label_tool.py
   ```
3. **准备数据集**:
   ```bash
   python scripts/prepare_yolo_data.py
   ```
4. **训练 YOLO 模型**:
   ```bash
   python scripts/train_yolo.py --epochs 30
   ```

---

## 性能指标

| 指标           | 数值   |
| -------------- | ------ |
| 检测准确率     | ~95%   |
| 平均跳跃成功率 | ~85%   |
| 单帧处理时间   | <100ms |
| 最高记录分数   | 3000+   |

---

## 许可证

MIT License
