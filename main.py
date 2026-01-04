"""
跳一跳游戏自动化脚本

基于 YOLO 目标检测的微信跳一跳游戏自动化方案。
"""

import time
from typing import Optional

import cv2
import numpy as np
import pyautogui
from pynput import keyboard

from src.screen_capture import ScreenCapture
from src.controller import MouseController
from src.calculator import JumpCalculator
from src.detector import Detector
from src.failure_collector import FailureCollector
from src import config

is_running = True


def on_press(key) -> None:
    """ESC 键退出监听回调"""
    global is_running
    if key == keyboard.Key.esc:
        print("\n[信号] 接收到退出指令 (ESC)...")
        is_running = False


def wait_for_stable(
    screen: ScreenCapture, threshold: float = 0.5, max_wait: float = 1.5
) -> Optional[np.ndarray]:
    """等待画面静止后返回稳定帧"""
    start_time = time.time()
    last_img = screen.capture(silent=True)
    if last_img is None:
        return None
        
    while time.time() - start_time < max_wait:
        time.sleep(0.05)
        curr_img = screen.capture(silent=True)
        if curr_img is None:
            continue

        diff = cv2.absdiff(curr_img, last_img)
        if np.mean(diff) < threshold:
            return curr_img
        last_img = curr_img

    return last_img


def main() -> None:
    """主循环入口"""
    global is_running

    print("=" * 60)
    print("   [跳一跳] 游戏自动化脚本 v1.0.0")
    print("=" * 60)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    screen = ScreenCapture()
    detector = Detector()
    calculator = JumpCalculator(press_coefficient=config.PRESS_COEFFICIENT)
    controller = MouseController()
    collector = FailureCollector()
    
    # 查找游戏窗口
    for _ in range(10):
        if not is_running:
            return
        if screen.find_window():
            break
        time.sleep(1)
    
    if screen.hwnd is None:
        print("[错误] 未找到游戏窗口")
        return
    
    screen.pop_window()
    print("[流程] 正在准备游戏状态...")

    # 跳过启动界面 (排行榜/开始按钮)
    for _ in range(5):
        if not is_running:
            return
        image = screen.capture()
        if image is None:
            time.sleep(0.5)
            continue
        
        # 检测并处理特殊界面
        if detector.is_ranking_list(image) and screen.game_region:
            h, w = image.shape[:2]
            pyautogui.click(screen.game_region[0] + 60, screen.game_region[1] + h - 60)
            time.sleep(1)
            continue
        
        start_pos = detector.get_start_game_button_pos(image)
        if start_pos and screen.game_region:
            pyautogui.click(screen.game_region[0] + start_pos[0], screen.game_region[1] + start_pos[1])
            time.sleep(1)
            continue
        
        # 正常游戏画面
        player_pos, _ = detector.detect(image)
        if player_pos:
            break
        time.sleep(0.5)
    
    if not is_running:
        return
    
    # 主循环
    print("\n" + "=" * 60)
    print("   [运行] YOLO 检测中 (ESC 停止)")
    print("=" * 60)
    
    if config.SHOW_DEBUG_WINDOW:
        cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("DEBUG", cv2.WND_PROP_TOPMOST, 1)
    
    jump_count = 0
    total_jumps = 0
    current_score = 0
    best_score = 0
    jump_history: list = []  # 最近2次跳跃状态 (image, player_pos, target_pos)
    
    try:
        while is_running:
            screen.pop_window(silent=True)
            time.sleep(0.01)  # 快速刷新周期
            
            # 等待画面静止后再检测
            image = wait_for_stable(screen)
            if image is None:
                continue
            
            if detector.is_game_over(image):
                # 尝试获取最终分数: 结算界面 -> 历史截图 -> 当前累计
                final_score = detector.get_final_score(image)
                if final_score <= 0 and jump_history:
                    for hist_img, _, _ in reversed(jump_history):
                        s = detector.get_score(hist_img)
                        if s > 0:
                            final_score = s
                            break
                if final_score <= 0:
                    final_score = current_score
                current_score = final_score

                for img, player, target in jump_history:
                    collector.save_failure(img, player, target)
                
                if current_score > best_score:
                    best_score = current_score
                    print(f"[游戏] 🎉 新纪录! 本轮: {jump_count}跳, 得分: {current_score}")
                else:
                    print(f"[游戏] 本轮: {jump_count}跳, 得分: {current_score} (最高: {best_score})")
                
                restart_pos = detector.get_restart_button_pos(image)
                if restart_pos and screen.game_region:
                    pyautogui.click(screen.game_region[0] + restart_pos[0], 
                                    screen.game_region[1] + restart_pos[1])
                    time.sleep(config.RESTART_WAIT_TIME + 0.5)
                    jump_count = 0
                    current_score = 0
                    jump_history.clear()
                    continue
                else:
                    time.sleep(1.0)
                    continue


            # 处理排行榜
            if detector.is_ranking_list(image) and screen.game_region:
                h, w = image.shape[:2]
                pyautogui.click(screen.game_region[0] + 60, screen.game_region[1] + h - 60)
                time.sleep(1.0)
                continue

            # 处理开始游戏
            start_pos = detector.get_start_game_button_pos(image)
            if start_pos and screen.game_region:
                pyautogui.click(screen.game_region[0] + start_pos[0], 
                                screen.game_region[1] + start_pos[1])
                time.sleep(1.0)
                continue

            # 检测目标
            player_pos, target_pos = detector.detect(image)
            
            if config.SHOW_DEBUG_WINDOW:
                debug_view = detector.visualize(image, player_pos, target_pos)
                h, w = image.shape[:2]
                scale = 600 / h
                cv2.imshow("DEBUG", cv2.resize(debug_view, (int(w * scale), 600)))
                cv2.waitKey(1)

            if player_pos is None or target_pos is None:
                continue

            # 计算并跳跃
            distance = calculator.calculate_distance(player_pos, target_pos)
            press_time = calculator.calculate_press_time(distance)
            
            jump_count += 1
            total_jumps += 1
            
            print(f"[跳跃] #{jump_count:03d} | 距离={distance:.0f}px | 按压={press_time/1000:.2f}s")
            
            # 记录跳跃前分数
            score_before = detector.get_score(image)
            if score_before < 0:
                score_before = current_score
            
            # 保存跳跃前状态（用于失败样本收集，最多保留2次）
            jump_history.append((image.copy(), player_pos, target_pos))
            if len(jump_history) > 2:
                jump_history.pop(0)
            
            # 执行跳跃
            controller.jump(int(press_time))
            
            # 等待结果
            time.sleep(0.6)
            check_image = screen.capture(silent=True)
            if check_image is not None:
                if detector.is_game_over(check_image):
                    continue  # 下一循环处理
                
                score_after = detector.get_score(check_image)
                if score_after > score_before:
                    delta = score_after - score_before
                    current_score = score_after
                    print(f"       ✓ SUCCESS +{delta} | 总分: {current_score}")
                
    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        listener.stop()
        if config.SHOW_DEBUG_WINDOW:
            cv2.destroyAllWindows()
    
    print(f"\n[统计] 共完成 {total_jumps} 次跳跃, 最高分: {best_score}")


if __name__ == "__main__":
    main()
