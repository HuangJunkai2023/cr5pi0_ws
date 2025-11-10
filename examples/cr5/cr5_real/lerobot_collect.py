#!/usr/bin/env python3
"""
使用 LeRobot 官方库进行 CR5 机械臂数据采集
Official LeRobot library for CR5 robot data collection
"""

import time
import numpy as np
from pathlib import Path
from datetime import datetime
from lerobot.record import LeRobotDataset
import pyrealsense2 as rs
import cv2
import sys
import threading
import shutil

# 添加 CR5 API 路径
cr5_path = Path(__file__).parent / "files"
sys.path.append(str(cr5_path))
from dobot_api import DobotApiDashboard, DobotApiFeedBack

# 尝试导入夹爪控制
try:
    from gripper import Robotiq2F85
    GRIPPER_AVAILABLE = True
except ImportError:
    GRIPPER_AVAILABLE = False
    print("⚠️  警告: 无法导入 gripper 模块，夹爪功能将不可用")

# 尝试导入键盘监听（Windows 特定）
try:
    import msvcrt
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠️  警告: 非 Windows 系统，键盘控制将使用替代方案")

# ====================
# 配置参数
# ====================

ROBOT_IP = "192.168.5.1"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 10  # 10 Hz 采集频率
TASK_NAME = "put the banana on the plate"  # 任务描述

# 夹爪配置
GRIPPER_PORT = "COM5"  # 夹爪串口号（请根据实际情况设置，例如 "COM5"、"COM3" 等）
                     # 设置为 None 则跳过夹爪初始化
GRIPPER_THRESHOLD_MM = 50.0  # 夹爪开口阈值（毫米）

# 全局变量
gripper = None
gripper_state = 0.0  # 0.0=打开, 1.0=关闭
gripper_lock = threading.Lock()
keyboard_thread_running = False

# ====================
# 夹爪控制类（异步线程版本）
# ====================

import queue

class GripperController:
    """夹爪控制器，支持 Robotiq 2F-85，使用异步线程避免阻塞"""
    
    def __init__(self, port=GRIPPER_PORT):
        global gripper, gripper_state
        self.gripper = None
        self.command_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        if not GRIPPER_AVAILABLE:
            print("⚠️  夹爪模块不可用，将使用模拟状态")
            return
        
        if port is None:
            print("ℹ️  夹爪端口未配置（GRIPPER_PORT = None），跳过夹爪初始化")
            print("    如需使用夹爪，请在代码中设置 GRIPPER_PORT = 'COM端口号'")
            return
        
        try:
            print(f"🤖 初始化夹爪 @ {port}...")
            self.gripper = Robotiq2F85(port)
            self.gripper.activate()  # 正确的方法名
            time.sleep(1)
            gripper = self.gripper
            gripper_state = 0.0
            print("✅ 夹爪已激活")
            
            # 启动工作线程
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            print("✅ 夹爪异步控制线程已启动")
            
        except Exception as e:
            print(f"⚠️  夹爪初始化失败: {e}")
            self.gripper = None
    
    def _worker(self):
        """夹爪控制工作线程"""
        global gripper_state
        
        while self.running:
            try:
                # 从队列获取命令，超时 0.1 秒
                command = self.command_queue.get(timeout=0.1)
                
                if command == "OPEN":
                    try:
                        # Robotiq 2F-85: close_gripper() 实际上是打开
                        self.gripper.close_gripper(speed=100, force=170)
                        with gripper_lock:
                            gripper_state = 0.0
                        print("🔓 夹爪打开")
                    except Exception as e:
                        print(f"⚠️  打开夹爪失败: {e}")
                
                elif command == "CLOSE":
                    try:
                        # Robotiq 2F-85: open_gripper() 实际上是关闭
                        self.gripper.open_gripper(speed=255, force=200, wait=0.1)
                        with gripper_lock:
                            gripper_state = 1.0
                        print("🔒 夹爪关闭")
                    except Exception as e:
                        print(f"⚠️  关闭夹爪失败: {e}")
                
                elif command == "STOP":
                    break
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  夹爪工作线程异常: {e}")
    
    def open(self):
        """打开夹爪（非阻塞）"""
        if self.gripper and self.running:
            self.command_queue.put("OPEN")
        elif not self.gripper:
            print("⚠️  夹爪未初始化，无法执行打开操作")
    
    def close(self):
        """关闭夹爪（非阻塞）"""
        if self.gripper and self.running:
            self.command_queue.put("CLOSE")
        elif not self.gripper:
            print("⚠️  夹爪未初始化，无法执行关闭操作")
    
    def get_state(self):
        """获取当前夹爪状态 (0.0=打开, 1.0=关闭)"""
        with gripper_lock:
            return gripper_state
    
    def cleanup(self):
        """清理资源"""
        if self.worker_thread and self.running:
            # 停止工作线程
            self.running = False
            self.command_queue.put("STOP")
            self.worker_thread.join(timeout=2)
            print("✅ 夹爪控制线程已停止")
        
        if self.gripper:
            try:
                self.gripper.close()
                print("✅ 夹爪已断开")
            except:
                pass

# ====================
# 键盘监听线程
# ====================

def keyboard_listener(gripper_controller):
    """键盘监听线程：监听夹爪控制按键"""
    global keyboard_thread_running
    
    print("\n" + "="*60)
    print("⌨️  键盘控制说明:")
    print("  按 'O' 键 → 打开夹爪 (Open)")
    print("  按 'C' 键 → 关闭夹爪 (Close)")
    print("  按 'Q' 键 → 结束当前 episode")
    print("="*60 + "\n")
    
    keyboard_thread_running = True
    
    if not KEYBOARD_AVAILABLE:
        print("⚠️  键盘监听不可用（非 Windows 系统）")
        return
    
    while keyboard_thread_running:
        try:
            if msvcrt.kbhit():
                key_bytes = msvcrt.getch()
                try:
                    key = key_bytes.decode('utf-8').upper()
                    
                    if key == 'O':  # 打开夹爪
                        gripper_controller.open()
                    elif key == 'C':  # 关闭夹爪
                        gripper_controller.close()
                        
                except UnicodeDecodeError:
                    pass
            
            time.sleep(0.05)
        except Exception as e:
            print(f"⚠️  键盘监听异常: {e}")
            break

# ====================
# CR5 机械臂控制
# ====================

class CR5Robot:
    def __init__(self, ip="192.168.5.1"):
        self.ip = ip
        self.dashboard = None
        self.feed = None
        self.is_enabled = False  # 跟踪使能状态
        
    def connect(self):
        print(f"🔌 连接 CR5 @ {self.ip}...")
        # DobotApi 构造函数会自动连接 socket
        self.dashboard = DobotApiDashboard(self.ip, 29999)
        self.feed = DobotApiFeedBack(self.ip, 30004)
        
        # 清除错误
        print("🧹 清除错误...")
        self.dashboard.ClearError()
        time.sleep(0.5)
        
        # 使能机器人
        print("⚡ 使能机器人...")
        result = self.dashboard.EnableRobot()
        print(f"   使能结果: {result}")
        self.is_enabled = True
        time.sleep(0.5)
        
        # 启用拖拽模式
        print("🤚 启用拖拽模式...")
        result = self.dashboard.StartDrag()
        print(f"   拖拽模式: {result}")
        print("✅ 已连接 (拖拽模式)")
        
    def disconnect(self):
        """安全断开连接，确保机器人失能"""
        print("\n🔌 断开 CR5 连接...")
        
        # 退出拖拽模式
        if self.dashboard:
            try:
                print("   停止拖拽...")
                self.dashboard.StopDrag()
            except Exception as e:
                print(f"   ⚠️  停止拖拽失败: {e}")
        
        # 失能机器人（重要！）
        if self.dashboard and self.is_enabled:
            try:
                print("   失能机器人...")
                self.dashboard.DisableRobot()
                self.is_enabled = False
                print("   ✅ 机器人已失能")
            except Exception as e:
                print(f"   ⚠️  失能失败: {e}")
        
        # 关闭连接
        if self.dashboard:
            try:
                self.dashboard.close()
            except Exception as e:
                print(f"   ⚠️  关闭 dashboard 失败: {e}")
        
        if self.feed:
            try:
                self.feed.close()
            except Exception as e:
                print(f"   ⚠️  关闭 feed 失败: {e}")
        
        print("✅ CR5 已断开")
        
    def get_state(self):
        """返回: (joint_positions, joint_velocities, gripper_position)"""
        try:
            feed_data = self.feed.feedBackData()
            if feed_data is None or len(feed_data) == 0:
                return None, None, None
            
            # 检查数据有效性
            if hex(feed_data['TestValue'][0]) != '0x123456789abcdef':
                return None, None, None
            
            # 关节位置（弧度）- QActual 是实际关节角度
            joints_deg = feed_data['QActual'][0]  # (6,) array
            joint_positions = np.deg2rad(joints_deg).astype(np.float32)
            
            # 关节速度（rad/s）- QDActual 是实际关节速度
            velocities_deg = feed_data['QDActual'][0]  # (6,) array
            joint_velocities = np.deg2rad(velocities_deg).astype(np.float32)
            
            # 夹爪（0=打开, 1=关闭）- 从 DigitalOutputs 获取
            digital_outputs = feed_data['DigitalOutputs'][0]
            gripper_closed = digital_outputs & 0x01  # 第一个 DO 位
            gripper_position = float(gripper_closed)
            
            return joint_positions, joint_velocities, gripper_position
        except Exception as e:
            print(f"⚠️  获取机械臂状态失败: {e}")
            return None, None, None

# ====================
# RealSense 相机
# ====================

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config = config
        
    def start(self):
        print("📷 启动相机（仅 RGB）...")
        self.pipeline.start(self.config)
        for _ in range(30):  # 预热
            self.pipeline.wait_for_frames()
        print("✅ 相机就绪")
        
    def stop(self):
        self.pipeline.stop()
        
    def get_frame(self):
        """返回 RGB 图像 (H, W, 3)"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image_bgr = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

# ====================
# 主采集函数
# ====================

def record_episode(dataset, robot, camera, gripper_controller, episode_idx, task_name):
    """
    记录一个 episode
    
    Args:
        dataset: LeRobotDataset 实例
        robot: CR5Robot 实例
        camera: RealSenseCamera 实例
        gripper_controller: GripperController 实例
        episode_idx: Episode 编号
        task_name: 任务名称
    """
    print(f"\n{'='*60}")
    print(f"📹 Episode {episode_idx} - {task_name}")
    print(f"{'='*60}")
    print("   按 's' 开始记录")
    print("   预览模式: 'O'=打开夹爪, 'C'=关闭夹爪")
    print("   按 'ESC' 或关闭窗口退出程序")
    
    # 预览并等待开始
    cv2.namedWindow('CR5 Data Collection')
    user_quit = False  # 标记用户是否要退出程序
    
    while True:
        image_rgb = camera.get_frame()
        if image_rgb is not None:
            display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 显示夹爪状态
            gripper_status = gripper_controller.get_state()
            gripper_text = "CLOSED" if gripper_status > 0.5 else "OPEN"
            gripper_color = (0, 0, 255) if gripper_status > 0.5 else (0, 255, 0)
            
            # 标题和提示信息 - 使用更小的字体
            cv2.putText(display, "=== PREVIEW MODE ===", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"Gripper: {gripper_text}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gripper_color, 1)
            
            # 键位说明
            cv2.putText(display, "[S] Start Recording", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(display, "[O] Open Gripper", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(display, "[C] Close Gripper", (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(display, "[ESC] Exit Program", (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            cv2.imshow('CR5 Data Collection', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            break
        elif key == 27:  # ESC 键
            user_quit = True
            cv2.destroyAllWindows()
            return False  # 返回 False 表示用户要退出程序
        elif key == ord('o') or key == ord('O'):
            gripper_controller.open()
        elif key == ord('c') or key == ord('C'):
            gripper_controller.close()
    
    # 如果用户选择退出
    if user_quit:
        return False
    
    print("🎬 开始记录...")
    
    # 创建 episode buffer
    dataset.create_episode_buffer()
    
    frame_count = 0
    start_time = time.time()
    target_period = 1.0 / FPS  # 精确的帧间隔 (0.1秒 for 10Hz)
    next_frame_time = start_time  # 下一帧的目标时间
    
    # 用于计算实际 FPS
    last_fps_time = start_time
    fps_frame_count = 0
    actual_fps = 0.0
    
    while True:
        loop_start = time.time()
        
        # 获取机械臂状态
        joint_pos, joint_vel, _ = robot.get_state()  # 忽略机械臂的夹爪状态
        if joint_pos is None:
            print("⚠️  无法获取机械臂状态")
            time.sleep(0.01)
            continue
        
        # 获取真实夹爪状态
        gripper_state_value = gripper_controller.get_state()
        
        # 获取图像
        image_rgb = camera.get_frame()
        if image_rgb is None:
            print("⚠️  无法获取图像")
            time.sleep(0.01)
            continue
        
        # 构建帧数据
        frame_data = {
            "observation.state": joint_pos,  # (6,) float32 关节位置
            "observation.images.top": image_rgb,  # (H, W, 3) RGB
            "action": np.concatenate([joint_vel, [gripper_state_value]]).astype(np.float32),  # (7,) float32 速度+夹爪
        }
        
        # 使用相对于 start_time 的精确时间戳
        current_time = time.time()
        timestamp = current_time - start_time
        dataset.add_frame(frame_data, task=task_name, timestamp=timestamp)
        
        frame_count += 1
        fps_frame_count += 1
        
        # 每秒更新一次实际 FPS
        if current_time - last_fps_time >= 1.0:
            actual_fps = fps_frame_count / (current_time - last_fps_time)
            fps_frame_count = 0
            last_fps_time = current_time
        
        # 显示
        display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        elapsed = time.time() - loop_start
        
        # 夹爪状态
        gripper_text = "CLOSED" if gripper_state_value > 0.5 else "OPEN"
        gripper_color = (0, 0, 255) if gripper_state_value > 0.5 else (0, 255, 0)
        
        # 顶部信息栏 - 深色背景
        cv2.rectangle(display, (0, 0), (640, 165), (0, 0, 0), -1)
        cv2.rectangle(display, (0, 0), (640, 165), (100, 100, 100), 1)
        
        # 录制状态标题
        cv2.putText(display, "=== RECORDING ===", (10, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 帧计数和 FPS
        cv2.putText(display, f"Frame: {frame_count:05d}", (10, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # 显示实际 FPS,颜色根据精度变化
        fps_diff = abs(actual_fps - FPS)
        if fps_diff < 0.5:
            fps_color = (0, 255, 0)  # 绿色 - 精确
        elif fps_diff < 1.0:
            fps_color = (0, 255, 255)  # 黄色 - 可接受
        else:
            fps_color = (0, 0, 255)  # 红色 - 偏差大
        
        cv2.putText(display, f"FPS: {actual_fps:.2f}/{FPS}", (10, 58), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1)
        
        # 夹爪状态
        cv2.putText(display, f"Gripper: {gripper_text}", (10, 78), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, gripper_color, 1)
        
        # 分隔线
        cv2.line(display, (10, 88), (630, 88), (100, 100, 100), 1)
        
        # 键位说明 - 更紧凑
        cv2.putText(display, "Keys:", (10, 108), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, "[O] Open", (60, 108), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "[C] Close", (150, 108), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, "[Q] Finish Episode", (245, 108), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        # Episode 信息
        cv2.putText(display, f"Episode: {episode_idx}", (10, 128), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        cv2.putText(display, f"Task: {task_name}", (10, 148), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        cv2.imshow('CR5 Data Collection', display)
        
        # 检查退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o') or key == ord('O'):
            gripper_controller.open()
        elif key == ord('c') or key == ord('C'):
            gripper_controller.close()
        
        # 计算下一帧的目标时间
        next_frame_time += target_period
        
        # 精确睡眠到下一帧时间
        current_time = time.time()
        sleep_time = next_frame_time - current_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # 如果已经落后,输出警告但继续
            if sleep_time < -0.01:  # 落后超过 10ms
                print(f"⚠️  警告: 帧 {frame_count} 落后 {-sleep_time*1000:.1f}ms")
            # 重新同步到当前时间
            next_frame_time = time.time() + target_period
    
    cv2.destroyAllWindows()
    
    # 保存 episode
    print(f"💾 保存 Episode {episode_idx} ({frame_count} 帧)...")
    dataset.save_episode()
    
    print(f"✅ Episode {episode_idx} 已保存")
    return True

# ====================
# 主程序
# ====================

def get_unique_dataset_name(task_name, root_path):
    """生成唯一的数据集名称"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"cr5_{task_name}_{timestamp}"
    dataset_path = root_path / dataset_name
    
    # 如果已存在，添加序号
    counter = 1
    while dataset_path.exists():
        dataset_name = f"cr5_{task_name}_{timestamp}_{counter}"
        dataset_path = root_path / dataset_name
        counter += 1
        
        # 安全检查，避免无限循环
        if counter > 100:
            # 使用毫秒级时间戳
            timestamp_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dataset_name = f"cr5_{task_name}_{timestamp_ms}"
            dataset_path = root_path / dataset_name
            break
    
    return dataset_name

def main():
    # 设置数据集根目录
    dataset_root = Path("./lerobot_data")
    
    # 如果 lerobot_data 已存在，重命名为备份
    if dataset_root.exists():
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"lerobot_data_backup_{backup_timestamp}"
        backup_path = Path(f"./{backup_name}")
        
        print("="*60)
        print("⚠️  检测到 lerobot_data 目录已存在")
        print("="*60)
        print(f"   原目录: {dataset_root.absolute()}")
        print(f"   备份为: {backup_path.absolute()}")
        
        try:
            # 使用 shutil.move 代替 Path.rename，更可靠
            shutil.move(str(dataset_root), str(backup_path))
            print("✅ 已重命名为备份目录")
        except Exception as e:
            print(f"❌ 重命名失败: {e}")
            print("   可能原因:")
            print("   1. 文件夹被其他程序占用（如文件资源管理器、VS Code）")
            print("   2. 文件夹中有文件被锁定")
            print("   ")
            print("   解决方法:")
            print("   - 关闭所有打开该文件夹的程序")
            print("   - 或手动删除/重命名 lerobot_data 目录")
            print("   - 然后重新运行此脚本")
            raise
        print("="*60 + "\n")
    
    # 生成唯一的数据集名称
    dataset_name = get_unique_dataset_name(TASK_NAME, dataset_root)
    dataset_path = dataset_root / dataset_name
    
    print("="*60)
    print("🤖 CR5 LeRobot 官方数据采集")
    print("="*60)
    print(f"📊 数据集: {dataset_name}")
    print(f"📷 分辨率: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"⏱️  频率: {FPS} Hz")
    print(f"🎯 任务: {TASK_NAME}")
    print("="*60)
    
    # 初始化
    robot = CR5Robot(ROBOT_IP)
    camera = RealSenseCamera(CAMERA_WIDTH, CAMERA_HEIGHT)
    gripper_controller = GripperController(GRIPPER_PORT)
    
    # 启动键盘监听线程
    keyboard_thread = None
    if KEYBOARD_AVAILABLE and gripper_controller.gripper:
        keyboard_thread = threading.Thread(
            target=keyboard_listener, 
            args=(gripper_controller,),
            daemon=True
        )
        keyboard_thread.start()
    
    try:
        robot.connect()
        camera.start()
        
        # 定义 LeRobot 特征
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"joint_{i}_vel" for i in range(6)] + ["gripper"],
            },
        }
        
        # 创建数据集
        print("\n💾 创建 LeRobot 数据集...")
        print(f"   数据集名称: {dataset_name}")
        print(f"   根目录: {dataset_root}")
        print(f"   完整路径: {dataset_path}")
        
        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=FPS,
            features=features,
            root=str(dataset_root),
            robot_type="cr5",
            use_videos=True,
            tolerance_s=0.5,  # 增加时间戳容差，允许更大的采集抖动
        )
        print(f"✅ 数据集已创建")
        
        # 记录 episodes
        episode_idx = 0
        while True:
            success = record_episode(dataset, robot, camera, gripper_controller, episode_idx, TASK_NAME)
            if not success:
                print("\n⚠️  用户选择退出程序")
                break
            episode_idx += 1
            
            # 询问是否继续
            print(f"\n" + "="*60)
            print(f"✅ Episode {episode_idx} 已保存！")
            print(f"📊 当前已完成 {episode_idx} 个 episodes")
            print("="*60)
            response = input("\n继续记录下一个 episode? (y/n): ").lower()
            if response != 'y':
                print("📝 用户选择结束采集")
                break
        
        # 完成
        print("\n" + "="*60)
        print(f"✅ 采集完成！总共 {episode_idx} 个 episodes")
        print(f"📂 数据集路径: {dataset_root / dataset_name}")
        print("\n🚀 下一步训练:")
        print(f"    python scripts/train_pytorch.py \\")
        print(f"      --pretrained-checkpoint pi0_droid \\")
        print(f"      --dataset-repo-id {dataset_name}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠️  检测到 Ctrl+C，正在安全退出...")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 开始清理资源...")
        
        # 停止键盘线程
        global keyboard_thread_running
        keyboard_thread_running = False
        if keyboard_thread:
            try:
                keyboard_thread.join(timeout=1)
                print("   ✅ 键盘监听已停止")
            except:
                pass
        
        # 清理相机
        try:
            if camera and hasattr(camera, 'pipeline'):
                camera.stop()
                print("   ✅ 相机已关闭")
        except Exception as e:
            print(f"   ⚠️  关闭相机失败: {e}")
        
        # 清理机械臂（最重要！确保失能）
        try:
            if robot:
                robot.disconnect()  # 内部会失能
        except Exception as e:
            print(f"   ⚠️  断开机械臂失败: {e}")
        
        # 清理夹爪
        try:
            if gripper_controller:
                gripper_controller.cleanup()
                print("   ✅ 夹爪已关闭")
        except Exception as e:
            print(f"   ⚠️  关闭夹爪失败: {e}")
        
        print("\n" + "="*60)
        print("✅ 清理完成，程序已安全退出")
        print("="*60)

if __name__ == "__main__":
    main()
