"""
系数拟合模块

功能：
1. 线性回归拟合 press_coefficient
2. 可视化拟合结果
3. 保存拟合结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from typing import Optional, List, Any


class CoefficientFitter:
    """系数拟合器"""


    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        初始化
        
        参数:
            output_dir: 结果输出目录（用于保存图表和结果json）。如果不指定，将在加载第一个会话时自动设置。
        """
        self.output_dir = output_dir
        self.samples: List[dict[str, Any]] = []
        self.sessions: List[str] = []

    def load_session(self, session_dir: str) -> bool:
        """加载一个会话的元数据并合并样本"""
        metadata_path = os.path.join(session_dir, "metadata.json")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                new_samples = data.get("samples", [])
                
                # 标记样本来源（可选）
                for s in new_samples:
                    s["_source"] = os.path.basename(session_dir)
                    
                self.samples.extend(new_samples)
                self.sessions.append(session_dir)
                
            print(f"[加载] 成功加载 {len(new_samples)} 个样本来自: {session_dir}")
            
            # 如果未指定输出目录，默认使用第一个加载的会话目录
            if self.output_dir is None:
                self.output_dir = session_dir
                
            return True
        except FileNotFoundError:
            print(f"[警告] 找不到元数据: {metadata_path}")
            return False
        except Exception as e:
            print(f"[错误] 加载失败 {session_dir}: {e}")
            return False

    def fit(self) -> Optional[float]:
        """
        线性回归拟合系数
        
        返回:
            拟合得到的系数，失败返回 None
        """
        print("=" * 50)
        print(f"[系数拟合] 线性回归分析 (共 {len(self.sessions)} 个会话)")
        print("=" * 50)

        # 筛选有效样本
        valid = [s for s in self.samples if s.get("distance") is not None and s.get("press_time") is not None]

        if len(valid) < 3:
            print(f"[错误] 有效样本不足（当前: {len(valid)}，至少需要3个）")
            return None

        distances: List[float] = [float(s["distance"]) for s in valid]
        times: List[float] = [float(s["press_time"]) for s in valid]

        # 线性回归
        coeffs = np.polyfit(distances, times, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        print(f"\n[结果] 拟合系数 (K): {slope:.4f}")
        print(f"[结果] 拟合截距 (B): {intercept:.2f}")
        print(f"[公式] PressTime = {slope:.4f} * Distance + {intercept:.2f}")

        # 显示样本详情
        # print("\n[数据] 样本详情:")
        # for i, s in enumerate(valid):
        #     predicted = slope * float(s["distance"]) + intercept
        #     error = float(s["press_time"]) - predicted
        #     print(f"  {i+1}. [{s.get('_source', '')}] 距离={s['distance']:.0f}px 时间={s['press_time']:.0f}ms 误差={error:+.0f}ms")

        # 计算平均误差
        errors = [abs(float(s["press_time"]) - (slope * float(s["distance"]) + intercept)) for s in valid]
        avg_error = sum(errors) / len(errors)
        print(f"\n[统计] 平均绝对误差: {avg_error:.2f} ms")

        # 可视化
        if self.output_dir:
            self._plot_fit(distances, times, slope, intercept)

        # 保存结果
        if self.output_dir:
            result = {
                "coefficient": slope,
                "intercept": intercept,
                "sample_count": len(valid),
                "sessions": self.sessions,
                "avg_error": avg_error
            }
            result_path = os.path.join(self.output_dir, "fit_result.json")
            try:
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"[保存] 结果已写入: {result_path}")
            except Exception as e:
                print(f"[警告] 结果保存失败: {e}")

        print(f"\n[建议] 请在 config.py 中更新: PRESS_COEFFICIENT = {slope:.4f}")

        return slope

    def _plot_fit(
        self, distances: List[float], times: List[float], slope: float, intercept: float
    ) -> None:
        """绘制拟合图"""
        try:
            import matplotlib.pyplot as plt
            
            # 设置中文字体（尝试一些常见的中文字体，如果只有英文环境则忽略）
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif'] 
            plt.rcParams['axes.unicode_minus'] = False

            plt.figure(figsize=(10, 6))

            # 散点
            plt.scatter(distances, times, color='blue', alpha=0.6, s=50, label='Samples', zorder=5)

            # 拟合线
            # 扩展一下 x 轴范围画线
            min_d, max_d = min(distances), max(distances)
            x_line = np.array([min_d - 20, max_d + 20])
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color='red', linewidth=2,
                    label=f'Fit: T = {slope:.4f} * D + {intercept:.2f}')

            plt.xlabel('Distance (px)', fontsize=12)
            plt.ylabel('Press Time (ms)', fontsize=12)
            plt.title(f'Jump Coefficient Fit (K={slope:.4f}) - {len(distances)} samples', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 保存
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, "fit_plot.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"\n[图表] 已保存到: {plot_path}")

            plt.show()

        except ImportError:
            print("\n[提示] 安装 matplotlib 可查看拟合图")
        except Exception as e:
            print(f"\n[警告] 绘图失败: {e}")


def list_sessions(base_dir: str) -> List[str]:
    """列出所有可用的会话目录"""
    if not os.path.exists(base_dir):
        return []
    
    sessions = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "metadata.json")):
            sessions.append(d)
    
    # 按名称（时间戳）倒序排列
    sessions.sort(reverse=True)
    return sessions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="系数拟合")
    parser.add_argument("-d", "--dir", type=str, help="指定会话目录（可选）")
    args = parser.parse_args()

    # 1. 尝试使用命令行参数
    if args.dir:
        fitter = CoefficientFitter()
        if fitter.load_session(args.dir):
            fitter.fit()
        else:
            print("[错误] 指定的目录无效或无数据")
    
    # 2. 进入交互式选择模式
    else:
        calibration_root = "./data/calibration"
        sessions = list_sessions(calibration_root)
        
        if not sessions:
            print(f"[提示] '{calibration_root}' 目录下没有找到有效的数据会话。")
            sys.exit(0)
            
        print(f"\n发现 {len(sessions)} 个历史数据会话:")
        for i, s in enumerate(sessions):
            print(f"  [{i+1}] {s}")
        
        print("\n请输入序号选择（多选请用逗号/空格分隔，例如 '1,3'，输入 'all' 选择所有）")
        selection = input(">>> ").strip().lower()
        
        selected_sessions = []
        if selection == 'all':
            selected_sessions = sessions
        else:
            # 处理 "1, 2" 或 "1 2" 格式
            parts = selection.replace(',', ' ').split()
            for p in parts:
                try:
                    idx = int(p) - 1
                    if 0 <= idx < len(sessions):
                        selected_sessions.append(sessions[idx])
                except ValueError:
                    pass
        
        if not selected_sessions:
            print("[退出] 未选择任何会话")
            sys.exit(0)
            
        print(f"\n[任务] 开始拟合 {len(selected_sessions)} 个会话的数据...")
        
        fitter = CoefficientFitter()
        for s_name in selected_sessions:
            full_path = os.path.join(calibration_root, s_name)
            fitter.load_session(full_path)
            
        if fitter.samples:
            fitter.fit()
        else:
            print("[错误] 没有加载到任何有效样本")
