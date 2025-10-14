#!/usr/bin/env python3
"""越疆 CR5 机械臂 + RealSense D415 + pi0 策略服务器的闭环抓取控制程序

本程序连接到运行中的 pi0 策略服务器（WebSocket），从手腕安装的 Intel RealSense D415
相机获取 RGB 图像，构建 pi0 策略所需的观测数据，并将策略输出的关节动作和夹爪指令
通过 TCP/IP SDK 发送到越疆 CR5 机械臂。

使用方法（在 OpenPI 工作空间内运行）：
# 启动策略服务器
uv run scripts/serve_policy.py --env DROID
# 启动 CR5 控制程序
    uv run --active examples/cr5_pi0_grasp.py \
        --robot-ip 192.168.5.1 \
        --server-host 127.0.0.1 \
        --prompt "抓取物体"

前置条件
-------------
1. pi0 策略服务器必须已经在运行，例如：
       uv run scripts/serve_policy.py --env DROID
   或者指向您 CR5 微调模型的自定义 checkpoint
2. RealSense D415 相机已安装在手腕处，并可通过 pyrealsense2 访问
   （如需安装：uv pip install pyrealsense2）
3. 越疆 TCP/IP Python SDK 已放置在 src/TCP-IP-Python-V4 目录下

注意事项
-------------
- CR5 机械臂必须处于 TCP 控制模式（非示教模式）
- 夹爪需连接到机械臂工具端数字输出口（默认 DO1）
- 关节限位、速度/加速度比例、delta_scale 等参数需根据实际场景调整
- 建议先在安全区域测试，确认运动范围和夹爪动作正确
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

try:  # External dependencies that are optional in unit tests.
    import pyrealsense2 as rs  # type: ignore
except ImportError as exc:  # pragma: no cover - hardware-specific.
    raise ImportError(
        "pyrealsense2 is required to stream images from the D415 camera. "
        "Install it with `pip install pyrealsense2`."
    ) from exc

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Reuse the existing DOBOT TCP/IP SDK that lives in the src tree. The folder name contains
# hyphens so we cannot import it via the regular dotted module syntax.
import importlib.util
from pathlib import Path


def _load_dobot_sdk() -> tuple[type, type]:
    sdk_dir = Path(__file__).resolve().parents[1] / "src" / "TCP-IP-Python-V4"
    sdk_file = sdk_dir / "dobot_api.py"
    if not sdk_file.exists():  # pragma: no cover - hardware deployment guard.
        raise FileNotFoundError(
            f"Expected DOBOT SDK at {sdk_file}. Please ensure the TCP-IP-Python-V4 folder "
            "is present in the src directory."
        )

    spec = importlib.util.spec_from_file_location("dobot_api", sdk_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for DOBOT SDK at {sdk_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        dashboard_cls = module.DobotApiDashboard
        feedback_cls = module.DobotApiFeedBack
    except AttributeError as exc:  # pragma: no cover - sanity check.
        raise ImportError(
            "DOBOT SDK does not expose DobotApiDashboard/DobotApiFeedBack as expected."
        ) from exc

    return dashboard_cls, feedback_cls


DobotApiDashboard, DobotApiFeedBack = _load_dobot_sdk()


RAD_TO_DEG = 180.0 / math.pi


@dataclasses.dataclass
class Args:
    """命令行参数配置"""

    robot_ip: str = "192.168.5.1"
    """CR5 控制器的 IP 地址（默认 192.168.5.1）"""

    server_host: str = "127.0.0.1"
    """pi0 策略服务器的主机地址（默认本地）"""

    server_port: int = 8000
    """pi0 策略服务器的端口号（默认 8000）"""

    prompt: str = "pick up the object"
    """发送给策略的任务指令提示词（如果策略支持自然语言输入）"""

    control_dt: float = 0.25
    """控制周期（秒）：每次策略查询之间的时间间隔，建议 0.1-0.5 秒"""

    joint_velocity_ratio: int = 30
    """关节速度比例 [1-100]：30 表示 30% 最大速度，建议测试时使用较小值（如 20-30）"""

    joint_acc_ratio: int = 20
    """关节加速度比例 [1-100]：20 表示 20% 最大加速度，建议测试时使用较小值"""

    action_mode: str = "delta"
    """动作模式：'delta' 表示增量模式（策略输出增量），'absolute' 表示绝对模式（策略输出目标位置）"""

    delta_scale: float = 0.15
    """Delta 模式下的缩放因子（弧度）：策略输出 [-1, 1] 会被缩放到 [-delta_scale, delta_scale] 弧度
    建议值：0.05-0.2，过大可能导致运动过快或不稳定"""

    joint_lower: tuple[float, ...] = (-2.97, -2.09, -2.97, -2.09, -2.97, -2.09)
    """关节下限（弧度）：CR5 官方限位约 ±170°(±2.97rad) 或 ±120°(±2.09rad)，根据实际情况调整"""
    
    joint_upper: tuple[float, ...] = (2.97, 2.09, 2.97, 2.09, 2.97, 2.09)
    """关节上限（弧度）：与 joint_lower 对应，保护机械臂不超出安全范围"""

    realsense_width: int = 640
    """RealSense D415 图像宽度（像素），默认 640"""
    
    realsense_height: int = 480
    """RealSense D415 图像高度（像素），默认 480"""
    
    realsense_serial: Optional[str] = None
    """RealSense 相机序列号（可选）：如果连接多个相机，可指定序列号选择特定相机"""

    use_depth: bool = False
    """是否启用深度图：True 表示同时捕获深度信息（当前版本策略未使用深度）"""

    env: str = "DROID"
    """策略环境名称：与 serve_policy.py 的 --env 参数对应（用于日志和调试）"""


class RealsenseCapture:
    """Context manager that streams color (and optional depth) frames from a RealSense sensor."""

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

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 (context manager protocol)
        self._pipeline.stop()

    def frames(self) -> Iterator[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Yields aligned color (and optionally depth) frames as numpy arrays."""
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


class CR5Pi0Controller:
    """Bridges the pi0 policy server, the CR5 robot, and the RealSense wrist camera."""

    def __init__(self, args: Args) -> None:
        self._args = args
        self._logger = logging.getLogger(self.__class__.__name__)

        # Connect to the policy server first so we fail fast if it is unavailable.
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host,
            port=args.server_port,
        )
        metadata = self._policy.get_server_metadata()
        self._logger.info("Connected to policy server: %s", metadata)

        # Connect to the CR5 dashboard (motion) and feedback sockets.
        self._dashboard = DobotApiDashboard(args.robot_ip, 29999)
        self._feedback = DobotApiFeedBack(args.robot_ip, 30004)

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------
    def enable_robot(self) -> None:
        """使能机械臂并设置速度比例"""
        self._logger.info("正在使能机械臂...")
        result = self._dashboard.EnableRobot()
        
        if "Not Tcp" in result:
            raise RuntimeError(
                "CR5 控制器未处于 TCP 模式。请在示教器上切换到 TCP 控制模式后重试。"
            )
        
        self._logger.info(f"机械臂使能成功: {result.strip()}")
        
        # 设置全局速度比例（可选，MovJ 命令中也可以单独指定）
        speed_result = self._dashboard.SpeedFactor(self._args.joint_velocity_ratio)
        self._logger.info(f"速度比例设置为 {self._args.joint_velocity_ratio}%: {speed_result.strip()}")

    def disable_robot(self) -> None:
        """停止机械臂运动并去使能（安全关闭）"""
        try:
            self._logger.info("正在停止机械臂运动...")
            self._dashboard.Stop()
            time.sleep(0.5)  # 等待停止命令执行
            
            self._logger.info("正在去使能机械臂...")
            self._dashboard.DisableRobot()
            self._logger.info("机械臂已安全关闭")
        except Exception:  # pragma: no cover - best effort cleanup.
            self._logger.exception("关闭机械臂时发生异常（尽力尝试）")

    def read_joint_state(self) -> np.ndarray:
        """读取当前关节位置（弧度）
        
        Returns:
            joint_positions_rad: 6个关节角度（弧度）
        """
        feedback = self._feedback.feedBackData()
        if feedback is None or len(feedback) == 0:
            raise RuntimeError("无法从 CR5 接收反馈数据包，请检查网络连接和机械臂状态")
        
        # CR5 的 QActual 返回的是弧度值（经测试确认）
        joints = np.asarray(feedback["QActual"][0][:6], dtype=np.float64)
        
        # 越疆 CR5 SDK 文档确认：QActual 返回弧度值
        # 但为了兼容性，如果数值明显超出 [-2π, 2π] 范围，则认为是角度制
        if np.max(np.abs(joints)) > 2 * math.pi + 0.5:
            self._logger.warning("检测到关节角度超出弧度范围，自动转换为弧度")
            joints = np.deg2rad(joints)
        
        return joints

    def command_joint_target(self, target_joints: np.ndarray) -> str:
        """发送关节目标位置到 CR5
        
        Args:
            target_joints: 6 个关节的目标角度（弧度）
            
        Returns:
            机械臂返回的命令执行结果字符串
        """
        # CR5 MovJ 的 coordinateMode=1 表示关节坐标模式，但输入需要是弧度值
        # 经过实际测试，CR5 SDK 在 coordinateMode=1 时接受弧度值
        target_rad = target_joints.tolist()
        
        result = self._dashboard.MovJ(
            *target_rad,
            coordinateMode=1,  # 关节坐标模式
            v=self._args.joint_velocity_ratio,
            a=self._args.joint_acc_ratio,
        )
        
        return result

    # ------------------------------------------------------------------
    # Policy interaction
    # ------------------------------------------------------------------
    def build_observation(
        self,
        joints: np.ndarray,
        color_bgr: np.ndarray,
    ) -> dict:
        """构建 pi0 策略所需的 DROID 风格观测字典
        
        Args:
            joints: 当前关节位置（弧度）
            color_bgr: RealSense 相机的 BGR 图像
            
        Returns:
            符合 pi0 输入格式的观测字典
        """
        # RealSense 返回 BGR 格式，需要转换为 RGB
        rgb = color_bgr[:, :, ::-1]  # BGR -> RGB
        
        # 调整图像大小到 pi0 期望的 224x224，保持宽高比并填充
        rgb = image_tools.resize_with_pad(rgb, 224, 224)
        rgb = image_tools.convert_to_uint8(rgb)
        
        # 构建符合 DROID 数据集格式的观测
        # 注意：由于只有手腕相机，将同一图像用于 exterior_image 和 wrist_image
        # 夹爪状态设为固定值（因为夹爪通过独立程序控制）
        obs = {
            "observation/exterior_image_1_left": rgb,  # 外部视角（使用手腕相机代替）
            "observation/wrist_image_left": rgb,        # 手腕相机视角
            "observation/joint_position": joints.astype(np.float32),
            "observation/gripper_position": np.asarray([0.0], dtype=np.float32),  # 固定值，不使用
            "prompt": self._args.prompt,
        }
        return obs

    def compute_joint_command(self, joints: np.ndarray, action: np.ndarray) -> np.ndarray:
        """根据策略输出的动作计算关节目标位置
        
        Args:
            joints: 当前关节位置（弧度）
            action: 策略输出的动作向量（至少包含 6 个关节动作 + 1 个夹爪动作）
            
        Returns:
            限制在安全范围内的目标关节位置（弧度）
        """
        if action.size < 6:
            raise ValueError(f"策略返回 {action.size} 个动作维度，至少需要 6 个（关节动作）")
        
        cmd = action[:6].astype(np.float64)
        
        if self._args.action_mode.lower() == "delta":
            # Delta 模式：策略输出增量，假设范围在 [-1, 1]
            # 先归一化到 [-1, 1]，再乘以 delta_scale 得到弧度增量
            delta = self._args.delta_scale * np.clip(cmd, -1.0, 1.0)
            cmd = joints + delta
            self._logger.debug(f"Delta 模式: 增量={delta}, 当前={joints}, 目标={cmd}")
        else:  # absolute 模式
            # Absolute 模式：策略直接输出目标关节位置
            self._logger.debug(f"Absolute 模式: 目标={cmd}")
        
        # 限制在 CR5 的关节限位范围内（安全保护）
        lower = np.asarray(self._args.joint_lower)
        upper = np.asarray(self._args.joint_upper)
        cmd_clipped = np.clip(cmd, lower, upper)
        
        if not np.allclose(cmd, cmd_clipped):
            self._logger.warning(f"目标关节位置被限制: {cmd} -> {cmd_clipped}")
        
        return cmd_clipped

    # ------------------------------------------------------------------
    # 主控制循环
    # ------------------------------------------------------------------
    def run(self, camera: RealsenseCapture) -> None:
        """主控制循环：相机捕获 -> 读取状态 -> 策略推理 -> 执行动作
        
        Args:
            camera: RealSense 相机捕获对象
        """
        self.enable_robot()
        frame_iterator = camera.frames()
        
        step_count = 0
        try:
            self._logger.info("=" * 60)
            self._logger.info("开始控制循环，目标任务: %s", self._args.prompt)
            self._logger.info("按 Ctrl+C 停止")
            self._logger.info("=" * 60)
            
            while True:
                loop_start = time.monotonic()
                
                # 1. 获取相机图像
                color, _ = next(frame_iterator)
                
                # 2. 读取当前机械臂状态
                joints = self.read_joint_state()
                
                # 3. 构建观测并查询策略
                observation = self.build_observation(joints, color)
                action_dict = self._policy.infer(observation)
                
                # 4. 解析动作输出
                actions = np.asarray(action_dict["actions"], dtype=np.float32)
                # 策略可能返回 [1, action_dim] 或 [action_dim]
                if actions.ndim == 2:
                    action_vec = actions[0]
                else:
                    action_vec = actions
                
                # 5. 计算关节目标
                joint_cmd = self.compute_joint_command(joints, action_vec)
                
                # 6. 发送命令到机械臂（不处理夹爪，夹爪通过独立程序控制）
                self._logger.info(
                    f"[步骤 {step_count}] 当前关节: {np.rad2deg(joints).round(1)} deg, "
                    f"目标关节: {np.rad2deg(joint_cmd).round(1)} deg"
                )
                
                move_result = self.command_joint_target(joint_cmd)
                self._logger.debug(f"MovJ 返回: {move_result.strip()}")
                
                # 注意：策略输出的第 7 维（action_vec[6]）是夹爪动作，
                # 但目前夹爪通过独立程序控制，此处不处理
                
                step_count += 1
                
                # 8. 保持控制频率
                elapsed = time.monotonic() - loop_start
                sleep_time = self._args.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self._logger.warning(f"控制周期超时: {elapsed:.3f}s > {self._args.control_dt}s")
                    
        finally:
            self._logger.info(f"控制循环结束，共执行 {step_count} 步")
            self.disable_robot()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    args = tyro.cli(Args)
    controller = CR5Pi0Controller(args)

    stop_requested = False

    def handle_sigint(signum, frame):  # noqa: ANN001 - required signature
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        with RealsenseCapture(
            width=args.realsense_width,
            height=args.realsense_height,
            serial=args.realsense_serial,
            enable_depth=args.use_depth,
        ) as camera:
            controller.run(camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down ...")
    except Exception as exc:
        logging.exception("Controller crashed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
