#!/usr/bin/env python3
"""越疆 CR5 机械臂 + RealSense D415 + 微调 pi0 策略的闭环控制程序

此程序连接到运行中的微调 pi0 策略服务器，从手腕安装的 Intel RealSense D415
相机获取 RGB 图像，构建策略所需的观测数据，并将策略输出的关节动作和夹爪指令
发送到越疆 CR5 机械臂。

使用方法：
# 1. 启动策略服务器（使用你的微调模型）
uv run scripts/serve_policy.py \
    --checkpoint /home/huang/learn_arm_robot/openpi/checkpoints/pi0_cr5_finetune_lora \
    --env DROID

# 2. 启动 CR5 控制程序
python examples/cr5/cr5_real_control.py \
    --robot-ip 192.168.5.1 \
    --server-host 127.0.0.1 \
    --prompt "抓取红色方块"

前置条件：
1. 微调的 pi0 策略服务器已在运行（监听 0.0.0.0:8000）
2. RealSense D415 相机已连接并可用
3. CR5 机械臂已上电且处于 TCP 控制模式
4. Robotiq 2F-85 夹爪已连接（可选，如果使用夹爪控制）
"""

from __future__ import annotations

import dataclasses
import logging
import math
import signal
import sys
import time
from collections.abc import Iterator
from typing import Optional

import numpy as np
import tyro

# 导入 RealSense SDK
try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise ImportError(
        "pyrealsense2 是必需的。安装方法：pip install pyrealsense2"
    ) from exc

# 导入 OpenPI 客户端库
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 导入本地的 DOBOT SDK 和夹爪控制（cr5_real 目录）
from pathlib import Path

# 将 cr5_real 目录添加到 sys.path
cr5_real_path = Path(__file__).parent / "cr5_real"
if str(cr5_real_path) not in sys.path:
    sys.path.insert(0, str(cr5_real_path))

# 导入 DOBOT API
try:
    from dobot_api import DobotApiDashboard, DobotApiFeedBack
except ImportError as exc:
    raise ImportError(
        f"无法导入 DOBOT API。请确保 {cr5_real_path}/dobot_api.py 存在"
    ) from exc

# 可选：导入 Robotiq 夹爪控制
_GRIPPER_IMPORT_ERROR = None
try:
    from gripper import Robotiq2F85
    _GRIPPER_AVAILABLE = True
except ImportError as exc:
    Robotiq2F85 = None
    _GRIPPER_AVAILABLE = False
    _GRIPPER_IMPORT_ERROR = str(exc)

RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


@dataclasses.dataclass
class Args:
    """命令行参数配置"""

    robot_ip: str = "192.168.5.1"
    """CR5 控制器的 IP 地址"""

    server_host: str = "127.0.0.1"
    """策略服务器的主机地址"""

    server_port: int = 8000
    """策略服务器的端口号"""

    prompt: str = "pick up the red cube and place it in the box"
    """任务提示词（发送给策略的自然语言指令）"""

    control_dt: float = 2.0
    """控制周期（秒），建议根据实际推理时间调整。Pi0 模型首次推理约 17s，后续约 1.5s"""

    joint_velocity_ratio: int = 30
    """关节速度比例 [1-100]，建议测试时使用 20-30"""

    joint_acc_ratio: int = 20
    """关节加速度比例 [1-100]，建议测试时使用 10-30"""

    action_mode: str = "delta"
    """动作模式：'delta'=增量模式, 'absolute'=绝对模式"""

    delta_scale: float = 0.15
    """Delta 模式的缩放因子（弧度），建议 0.05-0.2"""

    joint_lower: tuple[float, ...] = (-2.97, -2.09, -2.97, -2.09, -2.97, -2.09)
    """关节下限（弧度）"""
    
    joint_upper: tuple[float, ...] = (2.97, 2.09, 2.97, 2.09, 2.97, 2.09)
    """关节上限（弧度）"""

    realsense_width: int = 640
    """RealSense 图像宽度"""
    
    realsense_height: int = 480
    """RealSense 图像高度"""
    
    realsense_serial: Optional[str] = None
    """RealSense 相机序列号（可选）"""

    use_depth: bool = False
    """是否启用深度图"""

    enable_gripper: bool = False
    """是否启用 Robotiq 2F-85 夹爪控制"""

    gripper_port: str = "COM5"
    """夹爪串口（Windows）或设备路径（Linux: /dev/ttyUSB0）"""

    gripper_threshold: float = 0.5
    """夹爪动作阈值：策略输出 > threshold 则闭合，<= threshold 则打开"""

    max_steps: int = 1000
    """最大控制步数（防止无限循环）"""

    dry_run: bool = True
    """DRY-RUN 模式：True=只打印不执行，False=实际控制机械臂（默认 True 保证安全）"""

    env: str = "DROID"
    """策略环境名称（与 serve_policy.py 的 --env 对应）"""


class RealsenseCapture:
    """RealSense 相机捕获器（上下文管理器）"""

    def __init__(self, width: int, height: int, serial: Optional[str], enable_depth: bool):
        self._width = width
        self._height = height
        self._serial = serial
        self._enable_depth = enable_depth
        self._pipeline = rs.pipeline()
        self._cfg = rs.config()
        
        if serial:
            self._cfg.enable_device(serial)
        
        self._cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        if enable_depth:
            self._cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        
        self._align = rs.align(rs.stream.color)

    def __enter__(self) -> "RealsenseCapture":
        self._pipeline.start(self._cfg)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._pipeline.stop()

    def frames(self) -> Iterator[tuple[np.ndarray, Optional[np.ndarray]]]:
        """生成对齐的彩色（和可选的深度）帧"""
        while True:
            frames = self._pipeline.wait_for_frames()
            if self._enable_depth:
                frames = self._align.process(frames)
            
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            color = np.asanyarray(color_frame.get_data())
            depth = None
            
            if self._enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data())
            
            yield color, depth


class CR5Controller:
    """CR5 机械臂 + 微调 pi0 策略 + RealSense 相机的闭环控制器"""

    def __init__(self, args: Args) -> None:
        self._args = args
        self._logger = logging.getLogger(self.__class__.__name__)
        self._last_valid_joints = None  # 缓存上次有效的关节位置

        # 1. 连接策略服务器
        self._logger.info("正在连接策略服务器 %s:%d ...", args.server_host, args.server_port)
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host,
            port=args.server_port,
        )
        metadata = self._policy.get_server_metadata()
        self._logger.info("✓ 策略服务器已连接: %s", metadata)

        # 2. 实机模式：先初始化夹爪，再连接机械臂
        #    这样可以确保夹爪通信正常后再进行机械臂操作
        self._gripper = None
        self._dashboard = None
        self._feedback = None

        if args.dry_run:
            self._logger.warning("⚠ DRY-RUN 模式：将只打印数据，不实际控制机械臂")
        else:
            # 2a. 如果启用夹爪，先初始化夹爪
            if args.enable_gripper:
                if not _GRIPPER_AVAILABLE or Robotiq2F85 is None:
                    self._logger.warning("⚠ 夹爪模块未找到，跳过夹爪初始化")
                    if _GRIPPER_IMPORT_ERROR:
                        self._logger.warning(f"   导入错误: {_GRIPPER_IMPORT_ERROR}")
                        if "serial" in _GRIPPER_IMPORT_ERROR.lower():
                            self._logger.warning("   请安装 pyserial: pip install pyserial")
                    self._logger.warning(f"   夹爪模块路径: {cr5_real_path / 'gripper.py'}")
                else:
                    try:
                        self._logger.info("正在连接 Robotiq 2F-85 夹爪 (%s) ...", args.gripper_port)
                        self._gripper = Robotiq2F85(port=args.gripper_port, debug=False)
                        self._logger.info("正在激活夹爪（需要几秒钟）...")
                        self._gripper.activate()
                        self._logger.info("✓ Robotiq 夹爪已激活")
                        time.sleep(1)  # 等待夹爪完全就绪
                    except Exception as exc:
                        self._logger.error("✗ 夹爪初始化失败: %s", exc)
                        raise RuntimeError(f"夹爪初始化失败，请检查串口连接: {exc}")

            # 2b. 连接 CR5 机械臂
            self._logger.info("正在连接 CR5 机械臂 %s ...", args.robot_ip)
            self._dashboard = DobotApiDashboard(args.robot_ip, 29999)
            self._feedback = DobotApiFeedBack(args.robot_ip, 30004)
            self._logger.info("✓ CR5 机械臂已连接")

        # 3. DRY-RUN 模式也可以启用夹爪（用于测试夹爪通信）
        if args.dry_run and args.enable_gripper and self._gripper is None:
            if _GRIPPER_AVAILABLE and Robotiq2F85 is not None:
                try:
                    self._logger.info("正在连接 Robotiq 2F-85 夹爪 (%s) [DRY-RUN 测试]...", args.gripper_port)
                    self._gripper = Robotiq2F85(port=args.gripper_port, debug=False)
                    self._gripper.activate()
                    self._logger.info("✓ Robotiq 夹爪已激活 [DRY-RUN 可测试夹爪]")
                except Exception as exc:
                    self._logger.warning("⚠ 夹爪初始化失败 [DRY-RUN]: %s", exc)

    # ------------------------------------------------------------------
    # 机械臂控制方法
    # ------------------------------------------------------------------
    def enable_robot(self) -> None:
        """使能机械臂并设置速度比例"""
        if self._args.dry_run:
            self._logger.info("✓ [DRY-RUN] 跳过使能机械臂")
            return
        
        # 先清除错误
        self._logger.info("清除机械臂错误...")
        self._dashboard.ClearError()
        time.sleep(0.5)
        
        self._logger.info("正在使能机械臂...")
        result = self._dashboard.EnableRobot()
        
        if "Not Tcp" in result:
            raise RuntimeError(
                "CR5 控制器未处于 TCP 模式。\n"
                "请在示教器上切换到 TCP 控制模式后重试。"
            )
        
        self._logger.info("✓ 机械臂使能成功")
        
        # 设置速度比例
        speed_result = self._dashboard.SpeedFactor(self._args.joint_velocity_ratio)
        self._logger.info("✓ 速度比例设置为 %d%%", self._args.joint_velocity_ratio)

    def disable_robot(self) -> None:
        """停止机械臂运动并去使能"""
        if self._args.dry_run:
            self._logger.info("✓ [DRY-RUN] 跳过去使能机械臂")
            return
        
        try:
            self._logger.info("正在停止机械臂...")
            self._dashboard.Stop()
            time.sleep(0.5)
            
            self._logger.info("正在去使能机械臂...")
            self._dashboard.DisableRobot()
            self._logger.info("✓ 机械臂已安全关闭")
        except Exception as exc:
            self._logger.exception("关闭机械臂时发生异常: %s", exc)

    def read_joint_state(self) -> np.ndarray:
        """读取当前关节位置（弧度）
        
        Returns:
            joint_positions_rad: 6 个关节角度（弧度）
        """
        if self._args.dry_run:
            # DRY-RUN 模式：返回模拟的关节位置（零位）
            return np.zeros(6, dtype=np.float64)
        
        feedback = self._feedback.feedBackData()
        if feedback is None or len(feedback) == 0:
            raise RuntimeError("无法从 CR5 接收反馈数据，请检查网络连接")
        
        # 首次调用时打印调试信息
        if not hasattr(self, '_first_read_done'):
            self._first_read_done = True
            self._logger.info("=== 反馈数据结构调试 ===")
            self._logger.info("feedback type: %s", type(feedback))
            self._logger.info("feedback shape: %s", feedback.shape if hasattr(feedback, 'shape') else 'N/A')
            self._logger.info("feedback['QActual'] type: %s", type(feedback['QActual']))
            self._logger.info("feedback['QActual'] shape: %s", feedback['QActual'].shape if hasattr(feedback['QActual'], 'shape') else 'N/A')
            self._logger.info("feedback['QActual'][0] type: %s", type(feedback['QActual'][0]))
            self._logger.info("feedback['QActual'][0]: %s", feedback['QActual'][0])
            self._logger.info("=========================")
        
        # 检查数据有效性（feedBackData 返回的是 numpy 结构化数组）
        # TestValue 应该是 0x123456789abcdef，但有时数据包不完整
        try:
            test_value = feedback['TestValue'][0]
            expected = 0x123456789abcdef
            if test_value != expected:
                # 数据无效，但尝试继续读取 QActual
                self._logger.debug("反馈数据 TestValue = %s (expected %s)", hex(test_value), hex(expected))
        except (KeyError, IndexError, TypeError) as e:
            self._logger.debug("无法验证反馈数据: %s", e)
        
        # CR5 的 QActual 返回角度制（度）
        # MyType 定义: ('QActual', np.float64, (6, ))
        # feedback 是一个长度为 1 的结构化数组
        # feedback[0] 或 feedback['QActual'][0] 都能访问数据
        try:
            # 直接访问：feedback 是结构化数组，feedback['QActual'] 返回所有记录的 QActual 字段
            # 由于 feedback 长度为 1，feedback['QActual'][0] 获取第一条（也是唯一一条）记录的 6 个关节数据
            joints_deg = feedback['QActual'][0]  # 这应该直接是一个长度为 6 的数组
            
            # 转换为标准 numpy 数组
            joints_deg = np.asarray(joints_deg, dtype=np.float64)
            
            # 检查关节角度是否合理（CR5 工作范围通常在 -360 到 360 度）
            if np.all(joints_deg == 0):
                # 数据全为 0，可能是数据包不完整
                if self._last_valid_joints is not None:
                    self._logger.debug("关节角度全为 0，使用上次缓存数据")
                    return self._last_valid_joints  # 返回上次有效的数据
                else:
                    self._logger.warning("关节角度全为 0，且无缓存数据")
                    return np.zeros(6, dtype=np.float64)
            elif np.any(np.abs(joints_deg) > 360):
                self._logger.warning("关节角度超出合理范围: %s deg", joints_deg)
                if self._last_valid_joints is not None:
                    self._logger.warning("使用上次缓存数据")
                    return self._last_valid_joints
                
        except (KeyError, IndexError, TypeError) as e:
            self._logger.error("无法读取关节角度: %s", e)
            raise RuntimeError(f"反馈数据格式错误: {e}")
        
        # 转换为弧度
        joints_rad = np.deg2rad(joints_deg)
        
        # 缓存有效数据
        self._last_valid_joints = joints_rad.copy()
        
        return joints_rad

    def command_joint_target(self, target_joints: np.ndarray) -> str:
        """发送关节目标位置到 CR5
        
        Args:
            target_joints: 6 个关节的目标角度（弧度）
            
        Returns:
            机械臂返回的命令执行结果
        """
        if self._args.dry_run:
            # DRY-RUN 模式：只打印不发送
            target_deg = np.rad2deg(target_joints).round(2)
            print(f"[DRY-RUN] MovJ 命令:")
            print(f"  关节目标 (deg): {target_deg}")
            print(f"  关节目标 (rad): {target_joints.round(4)}")
            print(f"  速度比例: {self._args.joint_velocity_ratio}%")
            print(f"  加速度比例: {self._args.joint_acc_ratio}%")
            return "[DRY-RUN] 命令已记录"
        
        # CR5 MovJ 需要角度制（度），将弧度转换为角度
        target_deg = np.rad2deg(target_joints)
        target_list = target_deg.tolist()
        
        result = self._dashboard.MovJ(
            *target_list,
            coordinateMode=1,  # 关节坐标模式
            v=self._args.joint_velocity_ratio,
            a=self._args.joint_acc_ratio,
        )
        
        return result

    # ------------------------------------------------------------------
    # 夹爪控制方法
    # ------------------------------------------------------------------
    def command_gripper(self, gripper_action: float) -> None:
        """控制夹爪开合
        
        Args:
            gripper_action: 策略输出的夹爪动作值（通常 -1 到 1 或 0 到 1）
                          > threshold 则闭合，<= threshold 则打开
        """
        if self._gripper is None and not self._args.dry_run:
            return
        
        # 判断动作
        is_close = gripper_action > self._args.gripper_threshold
        action_str = "闭合" if is_close else "打开"
        
        if self._args.dry_run:
            # DRY-RUN 模式：只打印不执行
            print(f"[DRY-RUN] 夹爪命令: {action_str} (动作值: {gripper_action:.3f}, 阈值: {self._args.gripper_threshold})")
            return
        
        try:
            if is_close:
                # 闭合夹爪（抓取）
                self._logger.debug("闭合夹爪 (动作值: %.3f)", gripper_action)
                self._gripper.close_gripper(speed=100, force=170)
            else:
                # 打开夹爪（释放）
                self._logger.debug("打开夹爪 (动作值: %.3f)", gripper_action)
                self._gripper.open_gripper(speed=255, force=200, wait=1.0)
        except Exception as exc:
            self._logger.warning("夹爪控制失败: %s", exc)

    # ------------------------------------------------------------------
    # 策略交互方法
    # ------------------------------------------------------------------
    def build_observation(
        self,
        joints: np.ndarray,
        color_bgr: np.ndarray,
    ) -> dict:
        """构建策略所需的观测字典
        
        Args:
            joints: 当前关节位置（弧度）
            color_bgr: RealSense 的 BGR 图像
            
        Returns:
            符合策略输入格式的观测字典
        """
        # BGR -> RGB
        rgb = color_bgr[:, :, ::-1]
        
        # 调整图像到 224x224（策略期望的尺寸）
        rgb = image_tools.resize_with_pad(rgb, 224, 224)
        rgb = image_tools.convert_to_uint8(rgb)
        
        # 构建观测（CR5 策略格式）
        # CR5 策略期望: "image", "state", "prompt" (可选)
        # 根据训练时的数据采集，state 只包含 6 个关节位置（不包含夹爪）
        obs = {
            "image": rgb,
            "state": joints.astype(np.float32),  # 只有 6 个关节位置
            "prompt": self._args.prompt,
        }
        return obs

    def compute_joint_command(self, joints: np.ndarray, action: np.ndarray) -> np.ndarray:
        """根据策略输出计算关节目标位置
        
        Args:
            joints: 当前关节位置（弧度）
            action: 策略输出的动作向量（至少 6 维）
            
        Returns:
            限制在安全范围内的目标关节位置（弧度）
        """
        if action.size < 6:
            raise ValueError(f"策略返回 {action.size} 维动作，至少需要 6 维")
        
        cmd = action[:6].astype(np.float64)
        
        if self._args.action_mode.lower() == "delta":
            # Delta 模式：策略输出增量（-1 到 1）
            delta = self._args.delta_scale * np.clip(cmd, -1.0, 1.0)
            cmd = joints + delta
            self._logger.debug("Delta: 增量=%.3f, 当前=%.3f, 目标=%.3f", 
                             np.linalg.norm(delta), 
                             np.linalg.norm(joints), 
                             np.linalg.norm(cmd))
        else:
            # Absolute 模式：策略直接输出目标位置
            self._logger.debug("Absolute: 目标=%.3f", np.linalg.norm(cmd))
        
        # 限制在安全范围内
        lower = np.asarray(self._args.joint_lower)
        upper = np.asarray(self._args.joint_upper)
        cmd_clipped = np.clip(cmd, lower, upper)
        
        if not np.allclose(cmd, cmd_clipped):
            self._logger.warning("关节位置被限位: %s -> %s", 
                               np.rad2deg(cmd).round(1), 
                               np.rad2deg(cmd_clipped).round(1))
        
        return cmd_clipped

    # ------------------------------------------------------------------
    # 主控制循环
    # ------------------------------------------------------------------
    def run(self, camera: RealsenseCapture) -> None:
        """主控制循环
        
        流程：相机捕获 -> 读取状态 -> 策略推理 -> 执行动作
        
        Args:
            camera: RealSense 相机捕获对象
        """
        self.enable_robot()
        frame_iterator = camera.frames()
        
        # 等待反馈数据稳定（非 DRY-RUN 模式）
        if not self._args.dry_run:
            self._logger.info("等待机械臂反馈数据稳定...")
            
            # 先尝试清除可能存在的错误
            try:
                self._dashboard.ClearError()
                time.sleep(0.3)
            except Exception:
                pass
            
            for i in range(10):
                try:
                    joints = self.read_joint_state()
                    if not np.all(joints == 0):
                        self._logger.info("✓ 反馈数据已就绪，当前关节位置: %s deg", 
                                        np.rad2deg(joints).round(1))
                        break
                except Exception as e:
                    self._logger.debug("尝试 %d/10: %s", i+1, e)
                time.sleep(0.5)
            else:
                self._logger.warning("⚠ 反馈数据可能未就绪，但继续执行...")
        
        step_count = 0
        try:
            self._logger.info("=" * 70)
            self._logger.info("开始控制循环")
            self._logger.info("任务提示: %s", self._args.prompt)
            self._logger.info("控制频率: %.2f Hz (dt=%.3f s)", 1.0/self._args.control_dt, self._args.control_dt)
            self._logger.info("最大步数: %d", self._args.max_steps)
            self._logger.info("按 Ctrl+C 停止")
            self._logger.info("=" * 70)
            
            while step_count < self._args.max_steps:
                loop_start = time.monotonic()
                
                # 1. 获取相机图像
                color, _ = next(frame_iterator)
                
                # 2. 读取机械臂状态
                joints = self.read_joint_state()
                
                # 3. 构建观测并查询策略
                observation = self.build_observation(joints, color)
                action_dict = self._policy.infer(observation)
                
                # 4. 解析动作输出
                actions = np.asarray(action_dict["actions"], dtype=np.float32)
                if actions.ndim == 2:
                    action_vec = actions[0]
                else:
                    action_vec = actions
                
                # 5. 计算关节目标
                joint_cmd = self.compute_joint_command(joints, action_vec)
                
                # 6. 发送关节命令
                move_result = self.command_joint_target(joint_cmd)
                
                # 检查 MovJ 返回结果（仅非 DRY-RUN 模式）
                if not self._args.dry_run and move_result:
                    # ErrorId 含义:
                    # -30001: 机械臂未就绪/运动中
                    # -10000: 成功
                    # 其他: 各种错误
                    if "-30001" in str(move_result):
                        # -30001 通常表示上一个运动还没完成，这是正常的
                        # 不需要每次都打印警告
                        pass
                    elif "-10000" not in str(move_result):
                        # 其他错误才需要关注
                        self._logger.warning("MovJ 返回异常: %s", move_result)
                
                # 7. 控制夹爪（如果启用）
                if self._gripper is not None and action_vec.size >= 7:
                    gripper_action = action_vec[6]
                    self.command_gripper(gripper_action)
                
                # 8. 输出日志
                mode_prefix = "[DRY-RUN]" if self._args.dry_run else ""
                self._logger.info(
                    "%s [步骤 %4d] 当前: %s deg -> 目标: %s deg%s",
                    mode_prefix,
                    step_count,
                    np.rad2deg(joints).round(1),
                    np.rad2deg(joint_cmd).round(1),
                    f" | 夹爪: {action_vec[6]:.2f}" if action_vec.size >= 7 else ""
                )
                
                step_count += 1
                
                # 9. 保持控制频率
                elapsed = time.monotonic() - loop_start
                sleep_time = self._args.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self._logger.warning("控制周期超时: %.3f s > %.3f s", elapsed, self._args.control_dt)
                    
        except KeyboardInterrupt:
            self._logger.info("\n用户中断，停止控制循环")
        finally:
            self._logger.info("控制循环结束，共执行 %d 步", step_count)
            self.disable_robot()
            
            # 关闭夹爪连接
            if self._gripper is not None:
                try:
                    self._gripper.close()
                except Exception:
                    pass


def main() -> None:
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s"
    )
    
    args = tyro.cli(Args)
    
    # 打印配置信息
    logger = logging.getLogger("main")
    logger.info("=" * 70)
    logger.info("CR5 机械臂控制程序")
    logger.info("=" * 70)
    logger.info("配置:")
    logger.info("  机械臂 IP: %s", args.robot_ip)
    logger.info("  策略服务器: %s:%d", args.server_host, args.server_port)
    logger.info("  任务提示: %s", args.prompt)
    logger.info("  控制周期: %.3f s (%.1f Hz)", args.control_dt, 1.0/args.control_dt)
    logger.info("  动作模式: %s (scale=%.3f)", args.action_mode, args.delta_scale)
    logger.info("  速度/加速度比例: %d%% / %d%%", args.joint_velocity_ratio, args.joint_acc_ratio)
    logger.info("  启用夹爪: %s%s", args.enable_gripper, 
               f" ({args.gripper_port})" if args.enable_gripper else "")
    logger.info("  DRY-RUN 模式: %s", "✓ 开启（只打印不执行）" if args.dry_run else "✗ 关闭（实际控制）")
    logger.info("=" * 70)
    
    if args.dry_run:
        logger.warning("=" * 70)
        logger.warning("⚠ DRY-RUN 模式已启用 ⚠")
        logger.warning("  程序将只打印发送的数据，不会实际控制机械臂")
        logger.warning("  如需实际控制，请使用: --dry-run False")
        logger.warning("=" * 70)
    
    controller = CR5Controller(args)

    # 注册信号处理器
    def handle_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        with RealsenseCapture(
            width=args.realsense_width,
            height=args.realsense_height,
            serial=args.realsense_serial,
            enable_depth=args.use_depth,
        ) as camera:
            controller.run(camera)
    except KeyboardInterrupt:
        print("\n\n用户中断，正在安全关闭...")
    except Exception as exc:
        logging.exception("控制器崩溃: %s", exc)
    finally:
        # 确保退出时失能机械臂
        logger.info("=" * 70)
        logger.info("正在清理资源...")
        logger.info("=" * 70)
        try:
            controller.disable_robot()
        except Exception as e:
            logger.error("失能机械臂时出错: %s", e)
        
        logger.info("✓ 程序已安全退出")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
