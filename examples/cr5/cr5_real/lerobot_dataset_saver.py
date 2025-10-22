"""
标准LeRobot格式的数据集保存器实现
完全符合LeRobot官方库的数据格式标准
支持标准的目录结构、元数据格式和视频存储方式
"""

import json
import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class LeRobotDatasetSaver:
    """
    标准LeRobot格式的数据集保存器
    完全符合LeRobot官方库的数据格式和目录结构标准
    支持HuggingFace Hub的上传和下载
    """
    
    def __init__(self, repo_id: str, fps: int = 10, robot_type: str = "dobot_cr5"):
        """
        初始化数据集保存器
        
        Args:
            repo_id (str): 数据集仓库ID，例如 "dobot/cr5_teach_demo_20241009_143000"
            fps (int): 数据采集帧率
            robot_type (str): 机器人类型
        """
        self.repo_id = repo_id
        self.fps = fps
        self.robot_type = robot_type
        
        # 创建数据集目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dataset_name = f"lerobot_dataset_{timestamp}"
        self.root_path = Path(self.dataset_name)
        
        # 创建标准目录结构
        self._create_directory_structure()
        
        # 数据存储
        self.episodes_data = []
        self.frames_data = []
        self.current_episode_data = []
        self.current_episode_index = 0
        self.total_frames = 0
        self.current_task = None
        
        # 标准LeRobot特征定义（仅保留训练需要的数据）
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["J1", "J2", "J3", "J4", "J5", "J6"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),  # 6个关节速度(rad/s) + 1个夹爪位置
                "names": ["J1_vel", "J2_vel", "J3_vel", "J4_vel", "J5_vel", "J6_vel", "gripper"],
            },
            "observation.images.camera_color": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": {}
            },
            "observation.images.camera_depth": {
                "dtype": "video", 
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": {}
            },
            "episode_index": {
                "dtype": "int64",
                "shape": (),
                "names": None,
            },
            "frame_index": {
                "dtype": "int64", 
                "shape": (),
                "names": None,
            },
            "timestamp": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "next.done": {
                "dtype": "bool",
                "shape": (),
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": (),
                "names": None,
            },
        }
        
        # 视频编写器
        self.video_writers = {}
        self.video_frame_counts = {}
        
        # 统计信息
        self.stats = {key: {"min": None, "max": None, "mean": None, "std": None} 
                     for key in self.features.keys() 
                     if self.features[key]["dtype"] in ["float32", "int64"]}
        
        # 任务信息
        self.tasks = {}
        
        print(f"✓ 标准LeRobot数据集创建成功: {self.repo_id}")
        print(f"✓ 机器人类型: {self.robot_type}")
        print(f"✓ 采集频率: {self.fps} Hz")
        print(f"✓ 数据集路径: {self.root_path}")
    
    def _create_directory_structure(self):
        """创建标准LeRobot目录结构"""
        self.root_path.mkdir(parents=True, exist_ok=True)
        
        # 创建标准目录
        (self.root_path / "data").mkdir(exist_ok=True)
        (self.root_path / "meta").mkdir(exist_ok=True)
        (self.root_path / "videos").mkdir(exist_ok=True)
        
        # 创建视频子目录
        (self.root_path / "videos" / "observation.images.camera_color").mkdir(exist_ok=True, parents=True)
        (self.root_path / "videos" / "observation.images.camera_depth").mkdir(exist_ok=True, parents=True)
        
        print(f"✓ 创建标准LeRobot目录结构: {self.root_path}")
    
    def start_episode(self, task_name: str = "teach_drag"):
        """开始一个新的episode"""
        self.current_task = task_name
        self.current_episode_data = []
        
        # 添加任务到任务列表
        if task_name not in self.tasks:
            self.tasks[task_name] = len(self.tasks)
        
        # 初始化视频编写器
        self._init_video_writers()
        
        print(f"✓ 开始新的episode: {task_name}")
    
    def _init_video_writers(self):
        """初始化视频编写器"""
        # 使用更兼容的fourcc编码
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        
        # 创建视频文件路径
        color_video_dir = self.root_path / "videos" / "observation.images.camera_color"
        depth_video_dir = self.root_path / "videos" / "observation.images.camera_depth"
        
        color_path = color_video_dir / f"episode_{self.current_episode_index:06d}.mp4"
        depth_path = depth_video_dir / f"episode_{self.current_episode_index:06d}.mp4"
        
        # 彩色视频
        self.video_writers["color"] = cv2.VideoWriter(
            str(color_path), fourcc, self.fps, (640, 480)
        )
        
        # 深度视频
        self.video_writers["depth"] = cv2.VideoWriter(
            str(depth_path), fourcc, self.fps, (640, 480)
        )
        
        self.video_frame_counts = {"color": 0, "depth": 0}
    
    def add_frame(self, robot_data: Dict[str, Any], camera_data: Optional[Dict[str, Any]] = None):
        """添加一帧数据"""
        frame_index = len(self.current_episode_data)
        
        # 准备标准LeRobot格式的数据（角度制转弧度制 - pi0要求）
        observation_state = [
            np.deg2rad(float(robot_data['J1'])),
            np.deg2rad(float(robot_data['J2'])),
            np.deg2rad(float(robot_data['J3'])),
            np.deg2rad(float(robot_data['J4'])),
            np.deg2rad(float(robot_data['J5'])),
            np.deg2rad(float(robot_data['J6']))
        ]
        
        # 使用机器人实际速度（如果可用），否则将在episode结束时计算
        if all(key in robot_data for key in ['J1_vel', 'J2_vel', 'J3_vel', 'J4_vel', 'J5_vel', 'J6_vel']):
            # 使用实际速度（需要从角度/秒转换为弧度/秒）
            # 夹爪状态：从 robot_data 获取（0=打开，1=闭合）
            gripper_state = float(robot_data.get('gripper_state', 0.0))
            
            action = [
                np.deg2rad(float(robot_data['J1_vel'])),
                np.deg2rad(float(robot_data['J2_vel'])),
                np.deg2rad(float(robot_data['J3_vel'])),
                np.deg2rad(float(robot_data['J4_vel'])),
                np.deg2rad(float(robot_data['J5_vel'])),
                np.deg2rad(float(robot_data['J6_vel'])),
                gripper_state  # 夹爪状态：0=打开，1=闭合
            ]
        else:
            # 旧方法：将在episode结束时计算速度
            action = None
        
        # 准备帧数据（符合LeRobot标准格式）
        # 帧数据字典
        frame_data = {
            "observation.state": observation_state,
            "action": action,
            "episode_index": self.current_episode_index,
            "frame_index": frame_index,
            "timestamp": float(robot_data['timestamp']),
            "next.done": False,  # 将在episode结束时更新最后一帧
            "index": self.total_frames + frame_index,
        }
        
        # 处理视频数据（使用VideoFrame格式）
        if camera_data is not None:
            self._save_camera_frame(camera_data, frame_index)
            # 添加视频引用（使用标准VideoFrame格式）
            color_video_path = f"videos/observation.images.camera_color/episode_{self.current_episode_index:06d}.mp4"
            depth_video_path = f"videos/observation.images.camera_depth/episode_{self.current_episode_index:06d}.mp4"
            
            frame_data["observation.images.camera_color"] = {
                "path": color_video_path,
                "timestamp": float(frame_index / self.fps)
            }
            frame_data["observation.images.camera_depth"] = {
                "path": depth_video_path, 
                "timestamp": float(frame_index / self.fps)
            }
        else:
            # 保存空白帧
            self._save_empty_camera_frame(frame_index)
            # 仍然添加视频引用，但时间戳指向空白帧
            color_video_path = f"videos/observation.images.camera_color/episode_{self.current_episode_index:06d}.mp4"
            depth_video_path = f"videos/observation.images.camera_depth/episode_{self.current_episode_index:06d}.mp4"
            
            frame_data["observation.images.camera_color"] = {
                "path": color_video_path,
                "timestamp": float(frame_index / self.fps)
            }
            frame_data["observation.images.camera_depth"] = {
                "path": depth_video_path,
                "timestamp": float(frame_index / self.fps)
            }
        
        self.current_episode_data.append(frame_data)
        self.total_frames += 1
    
    def _save_camera_frame(self, camera_data: Dict[str, Any], frame_index: int):
        """保存相机帧到视频"""
        # 处理彩色图像
        if "color_image" in camera_data:
            color_img = camera_data["color_image"]
            # 确保数组是NumPy数组并且是正确的类型
            if not isinstance(color_img, np.ndarray):
                color_img = np.array(color_img)
            
            # 确保数据类型正确
            if color_img.dtype != np.uint8:
                color_img = color_img.astype(np.uint8)
            
            if color_img.shape != (480, 640, 3):
                color_img = cv2.resize(color_img, (640, 480))
            
            # 转换BGR到RGB (OpenCV使用BGR)
            if len(color_img.shape) == 3 and color_img.shape[2] == 3:
                color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            else:
                color_bgr = color_img
            
            self.video_writers["color"].write(color_bgr)
            self.video_frame_counts["color"] += 1
        
        # 处理深度图像
        if "depth_image" in camera_data:
            depth_img = camera_data["depth_image"]
            # 确保数组是NumPy数组
            if not isinstance(depth_img, np.ndarray):
                depth_img = np.array(depth_img)
            
            if len(depth_img.shape) == 2:
                # 将深度图转换为可视化的彩色图
                # 安全的除法处理
                max_val = depth_img.max()
                if max_val > 0:
                    depth_normalized = ((depth_img / max_val) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_img, dtype=np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
            else:
                depth_colored = depth_img.astype(np.uint8) if depth_img.dtype != np.uint8 else depth_img
            
            if depth_colored.shape[:2] != (480, 640):
                depth_colored = cv2.resize(depth_colored, (640, 480))
            
            self.video_writers["depth"].write(depth_colored)
            self.video_frame_counts["depth"] += 1
    
    def _save_empty_camera_frame(self, frame_index: int):
        """保存空白相机帧"""
        # 空白彩色帧
        empty_color = np.zeros((480, 640, 3), dtype=np.uint8)
        self.video_writers["color"].write(empty_color)
        self.video_frame_counts["color"] += 1
        
        # 空白深度帧
        empty_depth = np.zeros((480, 640, 3), dtype=np.uint8)
        self.video_writers["depth"].write(empty_depth)
        self.video_frame_counts["depth"] += 1
    
    def _compute_joint_velocities(self):
        """检查并填充缺失的动作数据"""
        missing_count = 0
        
        for i in range(len(self.current_episode_data)):
            # 如果动作数据为空，填充为0（不应该发生）
            if self.current_episode_data[i]["action"] is None:
                missing_count += 1
                self.current_episode_data[i]["action"] = [0.0] * 7
        
        if missing_count > 0:
            print(f"  ⚠ 警告: 有 {missing_count} 帧缺少速度数据，已填充为0")
        else:
            print(f"  ✓ 使用机器人实际速度数据（从TCP直接获取）")
    
    def save_episode(self):
        """保存当前episode"""
        if not self.current_episode_data:
            print("⚠ 当前episode没有数据，跳过保存")
            return
        
        # 计算关节速度（pi0要求）
        self._compute_joint_velocities()
        
        # 设置最后一帧的done标志
        if self.current_episode_data:
            self.current_episode_data[-1]["next.done"] = True
        
        # 关闭视频编写器
        for writer in self.video_writers.values():
            if writer:
                writer.release()
        
        # 计算episode统计信息
        episode_length = len(self.current_episode_data)
        episode_tasks = [self.current_task] if self.current_task else []
        
        # 保存episode元数据
        episode_metadata = {
            "episode_index": self.current_episode_index,
            "length": episode_length,
            "timestamp": datetime.now().isoformat(),
            "tasks": episode_tasks,
            "data/chunk_index": 0,  # 简化实现，所有数据放在一个chunk中
            "data/file_index": 0,
            "videos/observation.images.camera_color/chunk_index": 0,
            "videos/observation.images.camera_color/file_index": self.current_episode_index,
            "videos/observation.images.camera_color/from_timestamp": 0.0,
            "videos/observation.images.camera_color/to_timestamp": float((episode_length - 1) / self.fps),
            "videos/observation.images.camera_depth/chunk_index": 0,
            "videos/observation.images.camera_depth/file_index": self.current_episode_index,
            "videos/observation.images.camera_depth/from_timestamp": 0.0,
            "videos/observation.images.camera_depth/to_timestamp": float((episode_length - 1) / self.fps),
        }
        
        self.episodes_data.append(episode_metadata)
        
        # 将当前episode的帧数据添加到总数据中
        self.frames_data.extend(self.current_episode_data)
        
        # 更新统计信息
        self._update_stats()
        
        print(f"✓ Episode {self.current_episode_index} 已保存，包含 {len(self.current_episode_data)} 帧")
        
        # 准备下一个episode
        self.current_episode_index += 1
        self.current_episode_data = []
        self.video_writers = {}

    def _update_stats(self):
        """更新数据集统计信息"""
        if not self.current_episode_data:
            return
        
        # 更新数值特征的统计信息
        numeric_features = ["observation.state", "observation.cartesian_position", "action"]
        
        for feature in numeric_features:
            if feature in self.current_episode_data[0]:
                # 收集所有该特征的数据
                feature_data = np.array([frame[feature] for frame in self.current_episode_data])
                
                if self.stats[feature]["min"] is None:
                    # 首次计算
                    self.stats[feature]["min"] = feature_data.min(axis=0).tolist()
                    self.stats[feature]["max"] = feature_data.max(axis=0).tolist()
                    self.stats[feature]["mean"] = feature_data.mean(axis=0).tolist()
                    self.stats[feature]["std"] = feature_data.std(axis=0).tolist()
                else:
                    # 更新统计信息
                    current_min = np.array(self.stats[feature]["min"])
                    current_max = np.array(self.stats[feature]["max"])
                    
                    self.stats[feature]["min"] = np.minimum(current_min, feature_data.min(axis=0)).tolist()
                    self.stats[feature]["max"] = np.maximum(current_max, feature_data.max(axis=0)).tolist()
                    
                    # 简化的均值和标准差更新（这里可以改进为增量计算）
                    all_data = []
                    for frame in self.frames_data:
                        if feature in frame:
                            all_data.append(frame[feature])
                    if all_data:
                        all_data = np.array(all_data)
                        self.stats[feature]["mean"] = all_data.mean(axis=0).tolist()
                        self.stats[feature]["std"] = all_data.std(axis=0).tolist()
    
    def finalize_dataset(self) -> str:
        """完成数据集保存，生成标准LeRobot格式的元数据文件"""
        # 确保最后一个episode被保存
        if self.current_episode_data:
            self.save_episode()
        
        # 保存标准LeRobot元数据
        self._save_lerobot_metadata()
        
        # 保存帧数据为Parquet格式
        self._save_frame_data_parquet()
        
        print(f"✓ 标准LeRobot数据集保存完成")
        print(f"✓ 数据集路径: {self.root_path}")
        print(f"✓ 总Episodes: {len(self.episodes_data)}")
        print(f"✓ 总帧数: {self.total_frames}")
        
        return str(self.root_path)
    
    def _save_lerobot_metadata(self):
        """保存标准LeRobot元数据文件"""
        # 1. 保存 info.json
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes_data),
            "total_frames": self.total_frames,
            "fps": self.fps,
            "chunks_size": 100,  # 每个chunk的最大episode数
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self.features,
            "video": True,
        }
        
        with open(self.root_path / "meta" / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # 2. 保存 episodes.parquet
        episodes_df = pd.DataFrame(self.episodes_data)
        episodes_df.to_parquet(self.root_path / "meta" / "episodes.parquet", index=False)
        
        # 3. 保存 stats.json
        with open(self.root_path / "meta" / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # 4. 保存 tasks.parquet
        if self.tasks:
            tasks_data = [{"task": task, "task_index": idx} for task, idx in self.tasks.items()]
            tasks_df = pd.DataFrame(tasks_data)
            tasks_df.to_parquet(self.root_path / "meta" / "tasks.parquet", index=False)
        else:
            # 创建空的tasks文件
            tasks_df = pd.DataFrame(columns=["task", "task_index"])
            tasks_df.to_parquet(self.root_path / "meta" / "tasks.parquet", index=False)
        
        print(f"✓ 标准LeRobot元数据已保存")
    
    def _save_frame_data_parquet(self):
        """保存帧数据为Parquet格式"""
        if not self.frames_data:
            return
        
        try:
            # 确保数据目录存在
            data_dir = self.root_path / "data" / "chunk-000"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 将列表类型的数据展开为单独的列
            processed_frames = []
            for frame in self.frames_data:
                processed_frame = {}
                
                for key, value in frame.items():
                    if isinstance(value, list):
                        # 将列表展开为单独的列
                        if key == "observation.state":
                            for i, joint_name in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"]):
                                processed_frame[f"observation.state.{joint_name}"] = value[i]
                        elif key == "observation.cartesian_position":
                            for i, axis_name in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
                                processed_frame[f"observation.cartesian_position.{axis_name}"] = value[i]
                        elif key == "action":
                            for i, action_name in enumerate(["J1_vel", "J2_vel", "J3_vel", "J4_vel", "J5_vel", "J6_vel", "gripper"]):
                                processed_frame[f"action.{action_name}"] = value[i]
                        else:
                            # 其他列表类型数据，转为JSON字符串
                            processed_frame[key] = json.dumps(value)
                    elif isinstance(value, dict):
                        # 字典类型数据转为JSON字符串（如视频引用）
                        processed_frame[key] = json.dumps(value)
                    else:
                        # 标量数据直接保存
                        processed_frame[key] = value
                
                processed_frames.append(processed_frame)
            
            # 转换为DataFrame
            df = pd.DataFrame(processed_frames)
            
            # 确保数据类型正确
            df["episode_index"] = df["episode_index"].astype('int64')
            df["frame_index"] = df["frame_index"].astype('int64')
            df["timestamp"] = df["timestamp"].astype('float32')
            df["next.done"] = df["next.done"].astype('bool')
            df["index"] = df["index"].astype('int64')
            
            # 确保所有关节角度和速度列都是float32
            # 构建完整的DataFrame列
            joint_cols = [f"observation.state.{j}" for j in ["J1", "J2", "J3", "J4", "J5", "J6"]]
            action_cols = [f"action.{a}" for a in ["J1_vel", "J2_vel", "J3_vel", "J4_vel", "J5_vel", "J6_vel", "gripper"]]
            
            for col in joint_cols + action_cols:
                if col in df.columns:
                    df[col] = df[col].astype('float32')
            
            # 保存为Parquet格式（标准LeRobot格式）
            parquet_path = data_dir / "file-000000.parquet"
            df.to_parquet(parquet_path, index=False)
            
            print(f"✓ 帧数据已保存为Parquet格式: {parquet_path}")
            print(f"  包含列: {list(df.columns)}")
            print(f"  数据形状: {df.shape}")
            
        except Exception as e:
            print(f"⚠ Parquet保存失败: {e}")
            import traceback
            traceback.print_exc()
            # 紧急保存原始数据
            with open(self.root_path / "frames_data_raw.json", 'w', encoding='utf-8') as f:
                json.dump(self.frames_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"✓ 原始数据已保存: frames_data_raw.json")

    def validate_dataset(self) -> bool:
        """验证数据集格式的完整性"""
        try:
            # 检查必要的目录
            required_dirs = ["data", "meta", "videos"]
            for dir_name in required_dirs:
                if not (self.root_path / dir_name).exists():
                    print(f"❌ 缺少必要目录: {dir_name}")
                    return False
            
            # 检查必要的元数据文件
            required_meta_files = ["info.json", "episodes.parquet", "stats.json", "tasks.parquet"]
            for file_name in required_meta_files:
                if not (self.root_path / "meta" / file_name).exists():
                    print(f"❌ 缺少元数据文件: {file_name}")
                    return False
            
            # 检查数据文件
            data_files = list((self.root_path / "data").glob("**/*.parquet"))
            if not data_files:
                print("❌ 缺少数据文件")
                return False
            
            # 检查视频文件
            for episode in self.episodes_data:
                ep_idx = episode["episode_index"]
                
                # 检查彩色视频
                color_video_path = self.root_path / "videos" / "observation.images.camera_color" / f"episode_{ep_idx:06d}.mp4"
                if not color_video_path.exists():
                    print(f"❌ 缺少彩色视频文件: {color_video_path}")
                    return False
                
                # 检查深度视频
                depth_video_path = self.root_path / "videos" / "observation.images.camera_depth" / f"episode_{ep_idx:06d}.mp4"
                if not depth_video_path.exists():
                    print(f"❌ 缺少深度视频文件: {depth_video_path}")
                    return False
            
            print("✓ 数据集格式验证通过")
            return True
            
        except Exception as e:
            print(f"❌ 数据集验证失败: {e}")
            return False
    
    def create_dataset_card(self) -> str:
        """创建数据集README卡片"""
        card_content = f"""---
license: apache-2.0
tags:
- LeRobot
- robotics
- dobot
- teach-drag
task_categories:
- robotics
---

# {self.repo_id}

这是一个使用标准LeRobot格式的机器人数据集，包含{self.robot_type}机器人的示教轨迹数据。

## 数据集信息

- **机器人类型**: {self.robot_type}
- **采集频率**: {self.fps} Hz
- **总Episodes**: {len(self.episodes_data)}
- **总帧数**: {self.total_frames}
- **创建时间**: {datetime.now().isoformat()}

## 特征说明

{self._format_features_for_card()}

## 使用方法

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("{self.repo_id}")

# 获取一帧数据
frame = dataset[0]
print(frame.keys())
```

## 数据集结构

该数据集遵循标准的LeRobot格式，包含以下文件：

- `meta/info.json`: 数据集基本信息
- `meta/episodes.parquet`: Episode元数据
- `meta/stats.json`: 数据统计信息
- `meta/tasks.parquet`: 任务信息
- `data/`: Parquet格式的帧数据
- `videos/`: MP4格式的视频文件

创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = self.root_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        print(f"✓ 数据集README已创建: {readme_path}")
        return str(readme_path)
    
    def _format_features_for_card(self) -> str:
        """格式化特征信息用于README"""
        feature_lines = []
        for feature_name, feature_info in self.features.items():
            dtype = feature_info["dtype"]
            shape = feature_info["shape"]
            names = feature_info.get("names", "None")
            feature_lines.append(f"- **{feature_name}**: {dtype}, shape={shape}, names={names}")
        return "\n".join(feature_lines)


# 简化的工厂函数
def create_lerobot_saver(task_name: str = "teach_drag", fps: int = 10) -> LeRobotDatasetSaver:
    """
    创建标准LeRobot数据集保存器的便捷函数
    
    Args:
        task_name (str): 任务名称
        fps (int): 采集频率
        
    Returns:
        LeRobotDatasetSaver: 数据集保存器实例
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    repo_id = f"dobot/cr5_{task_name}_{timestamp}"
    
    saver = LeRobotDatasetSaver(
        repo_id=repo_id,
        fps=fps,
        robot_type="dobot_cr5"
    )
    
    saver.start_episode(task_name)
    return saver


def validate_lerobot_dataset(dataset_path: str) -> bool:
    """
    验证LeRobot数据集格式的完整性
    
    Args:
        dataset_path (str): 数据集路径
        
    Returns:
        bool: 验证是否通过
    """
    dataset_root = Path(dataset_path)
    
    try:
        # 检查基本目录结构
        required_dirs = ["data", "meta", "videos"]
        for dir_name in required_dirs:
            if not (dataset_root / dir_name).exists():
                print(f"❌ 缺少必要目录: {dir_name}")
                return False
        
        # 检查元数据文件
        required_meta_files = ["info.json", "episodes.parquet", "stats.json", "tasks.parquet"]
        for file_name in required_meta_files:
            if not (dataset_root / "meta" / file_name).exists():
                print(f"❌ 缺少元数据文件: {file_name}")
                return False
        
        # 验证info.json格式
        with open(dataset_root / "meta" / "info.json", 'r', encoding='utf-8') as f:
            info = json.load(f)
            required_info_keys = ["codebase_version", "robot_type", "total_episodes", 
                                "total_frames", "fps", "features"]
            for key in required_info_keys:
                if key not in info:
                    print(f"❌ info.json缺少必要字段: {key}")
                    return False
        
        print("✓ LeRobot数据集格式验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据集验证失败: {e}")
        return False