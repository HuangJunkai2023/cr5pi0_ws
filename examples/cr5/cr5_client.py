#!/usr/bin/env python3
"""
CR5 机械臂推理客户端
连接到 OpenPI 推理服务器，获取动作并控制 CR5 机械臂执行
"""

import dataclasses
import logging
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import pyrealsense2 as rs
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

# 添加 CR5 API 路径
cr5_test_path = Path(__file__).parent / "CR5_TCP_test"
sys.path.insert(0, str(cr5_test_path))

# 导入 CR5 API
try:
    from dobot_api import DobotApiDashboard, DobotApiFeedBack
except ImportError as e:
    print(f"❌ 无法导入 dobot_api: {e}")
    print(f"   请确保文件存在: {cr5_test_path / 'dobot_api.py'}")
    sys.exit(1)

# 尝试导入夹爪控制
try:
    from gripper import Robotiq2F85
    GRIPPER_AVAILABLE = True
except ImportError:
    GRIPPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  警告: 无法导入 gripper 模块，夹爪功能将不可用")

logger = logging.getLogger(__name__)


# ============================================================
# 配置参数
# ============================================================

# Episode 配置
MAX_STEPS_PER_EPISODE = 10000  # 每个 episode 的最大步数
EPISODE_TIMEOUT_SECONDS = None  # 每个 episode 的超时时间（秒），None 表示不限制

# 运动控制参数
SERVO_FREQUENCY = 30.0  # ServoJ 发送频率 (Hz)，建议 30 Hz（ServoJ 文档推荐 33Hz）
SERVO_DT = 1.0 / SERVO_FREQUENCY  # ServoJ 控制周期（秒）

# 推理频率（由于模型推理慢，通常 3-10 Hz）
INFERENCE_FREQUENCY = 6.0  # 推理请求频率 (Hz)
INFERENCE_DT = 1.0 / INFERENCE_FREQUENCY  # 推理周期（秒）

# 速度缩放因子（调整机械臂运动速度）
VELOCITY_SCALE = 1.0  # 范围 [0.1, 2.0]，适中速度

# 速度平滑参数（指数移动平均滤波）
VELOCITY_SMOOTHING = 0.2  # 范围 [0.0, 1.0]，适度平滑

# ============================================================


@dataclasses.dataclass
class Args:
    """命令行参数"""
    
    # 推理服务器配置
    host: str = "127.0.1.1"  # 推理服务器地址
    port: int = 8000  # 推理服务器端口
    api_key: str | None = None  # API key（可选）
    
    # CR5 机械臂配置
    robot_ip: str = "192.168.5.1"  # CR5 机械臂 IP 地址
    
    # 相机配置
    camera_width: int = 640  # 相机宽度
    camera_height: int = 480  # 相机高度
    camera_fps: int = 30  # 相机帧率
    
    # 夹爪配置
    gripper_port: str | None = "/dev/ttyUSB0"  # 夹爪串口，None 则跳过夹爪
    
    # 执行配置
    num_episodes: int = 10  # 执行的 episode 数量
    
    # 任务描述
    prompt: str = "put the flash drive on the book"  # 任务描述


class CR5Robot:
    """CR5 机械臂控制器"""
    
    def __init__(self, ip: str = "192.168.5.1"):
        self.ip = ip
        self.dashboard = None
        self.feed = None
        self.is_enabled = False
        
    def connect(self):
        """连接并初始化机械臂"""
        logger.info(f"🔌 连接 CR5 @ {self.ip}...")
        
        self.dashboard = DobotApiDashboard(self.ip, 29999)
        self.feed = DobotApiFeedBack(self.ip, 30004)
        
        # 清除错误（使能前必须先清除）
        logger.info("🧹 清除错误...")
        self.dashboard.ClearError()
        time.sleep(0.5)
        
        # 再次清除，确保错误完全清除
        self.dashboard.ClearError()
        time.sleep(0.5)
        
        # 使能机器人
        logger.info("⚡ 使能机器人...")
        result = self.dashboard.EnableRobot()
        logger.info(f"   使能结果: {result}")
        
        if "EnableRobot()" in str(result) or "OK" in str(result):
            self.is_enabled = True
            logger.info("   ✅ 使能成功")
        else:
            logger.warning(f"   ⚠️  使能可能失败，请检查机械臂状态")
            self.is_enabled = True  # 仍然设置为 True 以便后续可以尝试失能
        
        # 检查机械臂状态
        time.sleep(0.5)
        logger.info("🔍 检查机械臂状态...")
        try:
            feed_data = self.feed.feedBackData()
            if feed_data is not None and len(feed_data) > 0:
                robot_mode = feed_data['RobotMode'][0]
                error_status = feed_data['ErrorStatus'][0]
                enable_status = feed_data['EnableStatus'][0]
                
                logger.info(f"   机器人模式: {robot_mode}")
                logger.info(f"      (5=使能空闲, 7=运行中, 8=单次运动, 9=错误)")
                logger.info(f"   错误状态: {error_status}")
                logger.info(f"   使能状态: {enable_status}")
                
                if robot_mode == 9:
                    logger.warning("   ⚠️  机器人处于错误状态，ServoJ 可能无法工作")
                    logger.warning("   建议检查示教器并手动清除错误")
        except Exception as e:
            logger.warning(f"   ⚠️  无法获取状态: {e}")
        
        logger.info("✅ CR5 已连接并使能")
        
    def disconnect(self):
        """安全断开连接 - 必须失能机器人"""
        logger.info("\n🔌 断开 CR5 连接...")
        
        # 失能机器人（重要！退出时必须失能）
        if self.dashboard and self.is_enabled:
            try:
                logger.info("   失能机器人...")
                result = self.dashboard.DisableRobot()
                logger.info(f"   失能结果: {result}")
                self.is_enabled = False
                time.sleep(0.5)
                logger.info("   ✅ 机器人已失能")
            except Exception as e:
                logger.error(f"   ⚠️  失能失败: {e}")
                logger.error(f"   ⚠️  请手动检查机械臂状态！")
        elif self.dashboard and not self.is_enabled:
            logger.info("   机器人未使能，无需失能")
        
        # 关闭连接
        if self.dashboard:
            try:
                self.dashboard.close()
                logger.info("   Dashboard 已关闭")
            except Exception as e:
                logger.error(f"   ⚠️  关闭 dashboard 失败: {e}")
        
        if self.feed:
            try:
                self.feed.close()
                logger.info("   Feed 已关闭")
            except Exception as e:
                logger.error(f"   ⚠️  关闭 feed 失败: {e}")
        
        logger.info("✅ CR5 已断开")
        
    def get_state(self):
        """
        获取机械臂当前状态
        
        Returns:
            tuple: (joint_positions, joint_velocities, gripper_position)
                   joint_positions: (6,) 关节位置（弧度）
                   joint_velocities: (6,) 关节速度（rad/s）
                   gripper_position: float 夹爪位置 (0=打开, 1=关闭)
        """
        try:
            feed_data = self.feed.feedBackData()
            if feed_data is None or len(feed_data) == 0:
                logger.debug("Feed data is None or empty")
                return None, None, None
            
            # 检查机械臂错误状态（numpy 结构化数组）
            error_status = feed_data['ErrorStatus'][0]
            if error_status != 0:
                logger.debug(f"机械臂错误状态: {error_status}")
                # 尝试清除错误
                if self.dashboard:
                    self.dashboard.ClearError()
                    time.sleep(0.2)
                return None, None, None
            
            # 检查数据有效性
            test_value = hex(feed_data['TestValue'][0])
            if test_value != '0x123456789abcdef':
                logger.debug(f"Invalid TestValue: {test_value}")
                return None, None, None
            
            # 关节位置（弧度）
            joints_deg = feed_data['QActual'][0]  # (6,) array
            joint_positions = np.deg2rad(joints_deg).astype(np.float32)
            
            # 关节速度（rad/s）
            velocities_deg = feed_data['QDActual'][0]  # (6,) array
            joint_velocities = np.deg2rad(velocities_deg).astype(np.float32)
            
            # 夹爪状态（从数字输出获取）
            # 注意：DigitalOutputs 可能是浮点类型，需要先转换为整数
            digital_outputs = feed_data['DigitalOutputs'][0]
            digital_outputs_int = int(digital_outputs)
            gripper_closed = digital_outputs_int & 0x01
            gripper_position = float(gripper_closed)
            
            return joint_positions, joint_velocities, gripper_position
            
        except Exception as e:
            logger.error(f"⚠️  获取机械臂状态失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None, None
    
    def move_joints(self, joint_positions: np.ndarray) -> bool:
        """使用 ServoJ 实时伺服控制移动关节
        
        ServoJ 是实时控制指令，不需要队列模式，适合高频率连续控制。
        注意：使用默认的 t=0.1s, aheadtime=50, gain=500
        
        Args:
            joint_positions: 6个关节角度 (度)
            
        Returns:
            是否成功
        """
        if self.dashboard is None:
            logger.error("机械臂未连接")
            return False
        
        try:
            # 使用 ServoJ 进行实时伺服控制
            # ServoJ 不是队列指令，是实时指令，不会受 -30001 错误影响
            j1, j2, j3, j4, j5, j6 = joint_positions
            
            logger.debug(f"ServoJ: [{j1:.4f}, {j2:.4f}, {j3:.4f}, {j4:.4f}, {j5:.4f}, {j6:.4f}]")
            
            # ✅ ServoJ 只需要 6 个必选参数（关节角度）
            # 可选参数 t, aheadtime, gain 使用默认值即可
            cmd = f"ServoJ({j1:.6f},{j2:.6f},{j3:.6f},{j4:.6f},{j5:.6f},{j6:.6f})"
            
            result = self.dashboard.sendRecvMsg(cmd)
            
            # 只在有错误时打印详细日志
            if isinstance(result, str) and not result.startswith("0"):
                logger.error(f"❌ ServoJ 返回: {result}")
                logger.debug(f"发送的命令: {cmd}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"移动关节失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


class RealSenseCamera:
    """RealSense 相机控制器"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config = config
        self.width = width
        self.height = height
        
    def start(self):
        """启动相机"""
        logger.info("📷 启动相机...")
        self.pipeline.start(self.config)
        
        # 预热
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        logger.info("✅ 相机就绪")
        
    def stop(self):
        """停止相机"""
        self.pipeline.stop()
        
    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取一帧 RGB 图像
        
        Returns:
            np.ndarray: (H, W, 3) RGB 图像，uint8 格式
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
            
            # 转换为 numpy 数组 (H, W, 3) BGR
            color_image = np.asanyarray(color_frame.get_data())
            
            # 转换为 RGB
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            return rgb_image
            
        except Exception as e:
            logger.error(f"⚠️  获取图像失败: {e}")
            return None


class GripperController:
    """Robotiq 2F-85 夹爪控制器"""
    
    def __init__(self, port: Optional[str] = None):
        self.gripper = None
        self.current_state = 0.0  # 0.0=打开, 1.0=关闭
        
        if not GRIPPER_AVAILABLE:
            logger.warning("⚠️  夹爪模块不可用")
            return
        
        if port is None:
            logger.info("ℹ️  夹爪端口未配置，跳过夹爪初始化")
            return
        
        try:
            logger.info(f"🤖 初始化夹爪 @ {port}...")
            self.gripper = Robotiq2F85(port)
            self.gripper.activate()
            time.sleep(1)
            logger.info("✅ 夹爪已激活")
            
        except Exception as e:
            logger.error(f"⚠️  夹爪初始化失败: {e}")
            self.gripper = None
    
    def set_position(self, position: float):
        """
        设置夹爪位置
        
        Args:
            position: 目标位置 (0.0=打开, 1.0=关闭)
        """
        if self.gripper is None:
            return
        
        try:
            # 判断是打开还是关闭
            if position < 0.5 and self.current_state >= 0.5:
                # 打开夹爪
                self.gripper.close_gripper(speed=100, force=170)
                self.current_state = 0.0
                logger.debug("🔓 夹爪打开")
                
            elif position >= 0.5 and self.current_state < 0.5:
                # 关闭夹爪
                self.gripper.open_gripper(speed=255, force=200, wait=0.1)
                self.current_state = 1.0
                logger.debug("🔒 夹爪关闭")
                
        except Exception as e:
            logger.error(f"⚠️  设置夹爪位置失败: {e}")
    
    def get_state(self) -> float:
        """获取当前夹爪状态"""
        return self.current_state


def create_observation(
    robot: CR5Robot,
    camera: RealSenseCamera,
    prompt: str
) -> dict:
    """
    创建观测数据，用于发送给推理服务器
    
    Args:
        robot: CR5Robot 实例
        camera: RealSenseCamera 实例
        prompt: 任务描述
        
    Returns:
        dict: 观测数据字典
    """
    # 获取机械臂状态
    joint_pos, joint_vel, gripper_pos = robot.get_state()
    
    if joint_pos is None:
        logger.warning("⚠️  无法获取机械臂状态")
        # 返回零状态
        joint_pos = np.zeros(6, dtype=np.float32)
        joint_vel = np.zeros(6, dtype=np.float32)
        gripper_pos = 0.0
    
    # 状态向量只包含 6 个关节位置（不包含夹爪）
    # 这与训练数据格式一致：observation.state 是 (6,)
    # CR5 policy 会自动 padding 到 32 维
    state = joint_pos
    
    # 获取相机图像
    image = camera.get_frame()
    
    if image is None:
        logger.warning("⚠️  无法获取相机图像")
        # 返回黑色图像
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    
    # 转换图像格式为 (C, H, W)
    image_chw = np.transpose(image, (2, 0, 1))  # (3, H, W)
    
    # 构建观测字典 - CR5 格式
    # CR5 policy 期望: state (7维), image (单张图像), prompt (可选)
    observation = {
        "state": state,
        "image": image_chw,
        "prompt": prompt,
    }
    
    return observation


def execute_single_action(
    robot: CR5Robot,
    gripper: GripperController,
    joint_velocities: np.ndarray,
    gripper_position: float,
    prev_velocity: Optional[np.ndarray] = None,
    print_details: bool = False
) -> Optional[np.ndarray]:
    """
    执行单步动作
    
    Args:
        robot: CR5Robot 实例
        gripper: GripperController 实例
        joint_velocities: 6个关节速度 (rad/s)
        gripper_position: 夹爪位置 [0, 1]
        prev_velocity: 上一步的速度（用于平滑）
        print_details: 是否打印详细信息
        
    Returns:
        当前步的速度（用于下一步平滑）
    """
    # 速度平滑（指数移动平均滤波）
    if VELOCITY_SMOOTHING > 0 and prev_velocity is not None:
        joint_velocities_smooth = VELOCITY_SMOOTHING * joint_velocities + (1 - VELOCITY_SMOOTHING) * prev_velocity
    else:
        joint_velocities_smooth = joint_velocities
    
    # 执行关节运动
    current_pos, _, _ = robot.get_state()
    if current_pos is not None:
        # 欧拉积分: new_pos = current_pos + velocity * dt * scale
        target_pos_rad = current_pos + joint_velocities_smooth * SERVO_DT * VELOCITY_SCALE
        
        # ⚠️ 重要：ServoJ 需要度数，而 get_state() 返回弧度
        target_pos_deg = np.rad2deg(target_pos_rad)
        
        if print_details:
            logger.info(f"🔍 执行详情:")
            logger.info(f"   当前关节位置 (度): {np.rad2deg(current_pos)}")
            logger.info(f"   关节速度原始 (rad/s): {joint_velocities}")
            logger.info(f"   关节速度平滑后 (rad/s): {joint_velocities_smooth}")
            logger.info(f"   目标位置 (度): {target_pos_deg}")
            logger.info(f"   夹爪位置: {gripper_position:.4f}")
        
        # ServoJ 使用默认参数 (t=0.1s)
        success = robot.move_joints(target_pos_deg)
        
        if not success:
            logger.error("❌ 关节运动失败")
            return joint_velocities_smooth
    else:
        logger.warning("⚠️  无法获取当前关节位置，跳过本次动作")
        return joint_velocities_smooth
    
    # 执行夹爪动作
    gripper.set_position(float(gripper_position))
    
    # 返回当前速度用于下一步平滑
    return joint_velocities_smooth


def extract_actions(action: dict) -> Optional[np.ndarray]:
    """
    从推理结果中提取动作序列
    
    Args:
        action: 推理服务器返回的动作字典
        
    Returns:
        动作序列 (T, 7)，T 为时间步数，7 为 6 个关节速度 + 1 个夹爪位置
    """
    action_data = action.get("actions", None)
    
    if action_data is None:
        logger.warning("⚠️  动作数据为空")
        return None
    
    if not isinstance(action_data, np.ndarray):
        logger.warning(f"⚠️  动作类型不正确: {type(action_data)}")
        return None
    
    # 检查动作数据格式
    if action_data.ndim != 2 or action_data.shape[1] < 7:
        logger.warning(f"⚠️  动作维度不正确: {action_data.shape}")
        return None
    
    # action_data shape: (T, 7)
    # T: 时间步数（如 50）
    # 7: 6 个关节速度 + 1 个夹爪位置
    logger.info(f"📊 提取动作序列: shape={action_data.shape}, 时间步数={action_data.shape[0]}")
    
    return action_data


def main(args: Args) -> None:
    """主函数"""
    
    # 初始化各个组件
    logger.info("=" * 60)
    logger.info("CR5 机械臂推理客户端")
    logger.info("=" * 60)
    
    # 1. 连接推理服务器
    logger.info(f"🌐 连接推理服务器 @ {args.host}:{args.port}...")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"✅ 服务器元数据: {policy.get_server_metadata()}")
    
    # 2. 先初始化夹爪（在机械臂使能前）
    logger.info("\n" + "=" * 60)
    gripper = GripperController(port=args.gripper_port)
    logger.info("=" * 60)
    
    # 2.5 测试夹爪：执行一次夹取和释放
    logger.info("\n" + "=" * 60)
    logger.info("🤏 测试夹爪动作...")
    logger.info("   正在夹取（关闭）...")
    gripper.set_position(1.0)  # 夹取（闭合）
    time.sleep(1.5)  # 等待 1.5 秒
    logger.info("   正在释放（打开）...")
    gripper.set_position(0.0)  # 释放（打开）
    time.sleep(1.5)  # 等待 1.5 秒
    logger.info("✅ 夹爪测试完成")
    logger.info("=" * 60)
    
    # 3. 初始化机械臂（会清除错误并使能）
    logger.info("\n" + "=" * 60)
    robot = CR5Robot(ip=args.robot_ip)
    robot.connect()
    logger.info("=" * 60)
    
    # 4. 初始化相机
    logger.info("\n" + "=" * 60)
    camera = RealSenseCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps
    )
    camera.start()
    logger.info("=" * 60)
    
    try:
        # 预热推理服务器
        logger.info("🔥 预热推理服务器...")
        for _ in range(2):
            obs = create_observation(robot, camera, args.prompt)
            policy.infer(obs)
        logger.info("✅ 预热完成")
        
        # 执行多个 episode
        for episode_idx in range(args.num_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"📹 Episode {episode_idx + 1}/{args.num_episodes}")
            logger.info(f"{'='*60}")
            logger.info(f"任务: {args.prompt}")
            logger.info(f"最大步数: {MAX_STEPS_PER_EPISODE}")
            logger.info(f"ServoJ 频率: {SERVO_FREQUENCY} Hz")
            logger.info(f"推理频率: {INFERENCE_FREQUENCY} Hz")
            logger.info(f"速度缩放: {VELOCITY_SCALE}x")
            logger.info(f"速度平滑: {VELOCITY_SMOOTHING} (0=最平滑, 1=不滤波)")
            logger.info("按 Enter 开始执行，按 Ctrl+C 停止...")
            input()
            
            logger.info("🚀 开始执行...")
            step_count = 0
            episode_start_time = time.time()
            prev_velocity = None  # 用于速度平滑
            
            # 动作缓冲区
            action_buffer = None  # (T, 7) 动作序列
            action_index = 0  # 当前执行到第几步
            last_inference_time = 0  # 上次推理时间
            
            # 执行循环
            while True:
                step_start = time.time()
                
                # 检查是否超过最大步数
                if step_count >= MAX_STEPS_PER_EPISODE:
                    logger.info(f"✅ 达到最大步数 {MAX_STEPS_PER_EPISODE}，结束 episode")
                    break
                
                # 检查是否超时（如果设置了超时）
                if EPISODE_TIMEOUT_SECONDS is not None and time.time() - episode_start_time > EPISODE_TIMEOUT_SECONDS:
                    logger.warning(f"⏱️  超时 {EPISODE_TIMEOUT_SECONDS}s，结束 episode")
                    break
                
                # 1. 检查是否需要推理新动作
                # 策略：最高实时性 - 尽可能频繁推理以获取最新状态反馈
                # 条件：动作缓冲区空 或 到达推理时间间隔
                should_inference = (
                    action_buffer is None or
                    (time.time() - last_inference_time) >= INFERENCE_DT
                )
                
                if should_inference:
                    logger.info(f"🔄 请求推理 (step={step_count}, buffer_idx={action_index}/{len(action_buffer) if action_buffer is not None else 0})")
                    observation = create_observation(robot, camera, args.prompt)
                    action_dict = policy.infer(observation)
                    action_buffer = extract_actions(action_dict)
                    action_index = 0
                    last_inference_time = time.time()
                    
                    if action_buffer is not None:
                        logger.info(f"✅ 获得新动作序列: {action_buffer.shape[0]} 步")
                        if step_count == 0:
                            logger.info("=" * 60)
                            logger.info(f"📊 动作序列: shape={action_buffer.shape}")
                            logger.info(f"   时间步数: {action_buffer.shape[0]}")
                            logger.info(f"   每步维度: {action_buffer.shape[1]} (6关节+1夹爪)")
                            logger.info("=" * 60)
                
                # 2. 从缓冲区取出当前步的动作
                if action_buffer is None or action_index >= len(action_buffer):
                    logger.warning("⚠️  动作缓冲区耗尽")
                    break
                
                current_action = action_buffer[action_index]
                joint_velocities = current_action[:6]
                gripper_position = current_action[6]
                action_index += 1
                
                # 3. 执行动作
                print_details = (step_count == 0)
                prev_velocity = execute_single_action(
                    robot, gripper,
                    joint_velocities, gripper_position,
                    prev_velocity, print_details
                )
                
                step_count += 1
                
                # 4. 控制 ServoJ 频率 (30Hz)
                elapsed = time.time() - step_start
                if elapsed < SERVO_DT:
                    time.sleep(SERVO_DT - elapsed)
                
                # 显示进度（每 10 步）
                if step_count % 10 == 0:
                    elapsed = time.time() - episode_start_time
                    logger.info(f"   步数: {step_count}/{MAX_STEPS_PER_EPISODE}, 已用时: {elapsed:.1f}s")
            
            # Episode 结束统计
            episode_duration = time.time() - episode_start_time
            logger.info(f"✅ Episode {episode_idx + 1} 完成: {step_count} 步, 用时 {episode_duration:.1f}s")
        
        logger.info("\n🎉 所有 episode 执行完成！")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  用户中断")
        
    except Exception as e:
        logger.error(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源（按顺序：相机 -> 机械臂失能 -> 关闭连接）
        logger.info("\n" + "=" * 60)
        logger.info("🧹 清理资源...")
        logger.info("=" * 60)
        
        try:
            camera.stop()
            logger.info("✅ 相机已停止")
        except Exception as e:
            logger.error(f"⚠️  停止相机失败: {e}")
        
        try:
            robot.disconnect()  # 会自动失能机器人
        except Exception as e:
            logger.error(f"⚠️  断开机械臂失败: {e}")
        
        logger.info("=" * 60)
        logger.info("✅ 清理完成")
        logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main(tyro.cli(Args))
