import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
import json


class D415CameraRecorder:
    """Intel RealSense D415 深度相机录制类"""
    
    def __init__(self, width=640, height=480, fps=30):
        """
        初始化相机录制器
        
        Args:
            width (int): 图像宽度，默认640
            height (int): 图像高度，默认480
            fps (int): 帧率，默认30
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # 初始化RealSense管道
        self.pipeline = None
        self.config = None
        self.align = None
        self.is_recording = False
        
        # 存储路径
        self.base_folder = None
        self.depth_folder = None
        self.color_folder = None
        
        # 相机内参
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        
        # 记录计数器
        self.frame_count = 0
        
    def initialize_camera(self):
        """初始化相机连接"""
        try:
            # 检查是否有可用的RealSense设备
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("❌ 未找到任何RealSense设备")
                return False
            
            print(f"✓ 找到 {len(devices)} 个RealSense设备")
            
            # 创建管道
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # 配置深度流和彩色流
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # 启动管道
            profile = self.pipeline.start(self.config)
            
            # 创建对齐对象（将深度图对齐到彩色图）
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # 获取深度传感器的深度比例
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # 获取相机内参
            color_stream = profile.get_stream(rs.stream.color)
            depth_stream = profile.get_stream(rs.stream.depth)
            
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print("✓ D415 相机初始化成功")
            print(f"  - 分辨率: {self.width}x{self.height}")
            print(f"  - 帧率: {self.fps} FPS")
            print(f"  - 深度比例: {self.depth_scale}")
            
            return True
            
        except RuntimeError as e:
            if "No device connected" in str(e):
                print("❌ 相机未连接，请检查USB连接")
            elif "Permission denied" in str(e):
                print("❌ 相机权限被拒绝，请以管理员身份运行或检查设备权限")
            else:
                print(f"❌ 相机运行时错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 相机初始化失败: {e}")
            return False
    
    def setup_recording_folders(self, base_path):
        """设置录制文件夹"""
        try:
            # 创建基础文件夹
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.base_folder = os.path.join(base_path, f"camera_data_{timestamp}")
            
            # 创建子文件夹
            self.depth_folder = os.path.join(self.base_folder, "depth")
            self.color_folder = os.path.join(self.base_folder, "color")
            
            os.makedirs(self.depth_folder, exist_ok=True)
            os.makedirs(self.color_folder, exist_ok=True)
            
            print(f"✓ 录制文件夹创建成功: {self.base_folder}")
            return True
            
        except Exception as e:
            print(f"❌ 创建录制文件夹失败: {e}")
            return False
    
    def capture_frame(self, frame_index, timestamp):
        """
        捕获一帧数据
        
        Args:
            frame_index (int): 帧索引
            timestamp (float): 时间戳
            
        Returns:
            dict: 包含文件路径和相机数据的字典，失败返回None
        """
        if not self.pipeline or not self.is_recording:
            return None
            
        try:
            # 等待帧数据
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # 对齐帧
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 生成文件名
            frame_filename = f"frame_{frame_index:06d}_{timestamp:.3f}"
            depth_path = os.path.join(self.depth_folder, f"{frame_filename}.png")
            color_path = os.path.join(self.color_folder, f"{frame_filename}.jpg")
            
            # 保存图像
            cv2.imwrite(depth_path, depth_image)
            cv2.imwrite(color_path, color_image)
            
            # 计算深度统计信息
            valid_depth = depth_image[depth_image > 0]
            depth_stats = {
                'min_depth': float(np.min(valid_depth)) * self.depth_scale if len(valid_depth) > 0 else 0,
                'max_depth': float(np.max(valid_depth)) * self.depth_scale if len(valid_depth) > 0 else 0,
                'mean_depth': float(np.mean(valid_depth)) * self.depth_scale if len(valid_depth) > 0 else 0,
                'valid_pixels': int(len(valid_depth))
            }
            
            self.frame_count += 1
            
            return {
                'frame_index': frame_index,
                'timestamp': timestamp,
                'depth_image_path': os.path.relpath(depth_path, self.base_folder),
                'color_image_path': os.path.relpath(color_path, self.base_folder),
                'depth_stats': depth_stats,
                'camera_intrinsics': {
                    'color': {
                        'width': self.color_intrinsics.width,
                        'height': self.color_intrinsics.height,
                        'fx': self.color_intrinsics.fx,
                        'fy': self.color_intrinsics.fy,
                        'ppx': self.color_intrinsics.ppx,
                        'ppy': self.color_intrinsics.ppy
                    },
                    'depth': {
                        'width': self.depth_intrinsics.width,
                        'height': self.depth_intrinsics.height,
                        'fx': self.depth_intrinsics.fx,
                        'fy': self.depth_intrinsics.fy,
                        'ppx': self.depth_intrinsics.ppx,
                        'ppy': self.depth_intrinsics.ppy
                    },
                    'depth_scale': self.depth_scale
                }
            }
            
        except Exception as e:
            print(f"捕获帧数据失败: {e}")
            return None
        
    def capture_frame_with_data(self, frame_index, timestamp):
        """
        捕获一帧数据，同时返回图像数据和保存路径
        
        Args:
            frame_index (int): 帧索引
            timestamp (float): 时间戳
            
        Returns:
            dict: 包含文件路径、相机数据和图像numpy数组的字典，失败返回None
        """
        if not self.pipeline or not self.is_recording:
            return None
            
        try:
            # 等待帧数据
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # 对齐帧
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 生成文件名
            frame_filename = f"frame_{frame_index:06d}_{timestamp:.3f}"
            depth_path = os.path.join(self.depth_folder, f"{frame_filename}.png")
            color_path = os.path.join(self.color_folder, f"{frame_filename}.jpg")
            
            # 保存图像
            cv2.imwrite(depth_path, depth_image)
            cv2.imwrite(color_path, color_image)
            
            # 计算深度统计信息
            valid_depth = depth_image[depth_image > 0]
            depth_stats = {
                'min_depth': float(np.min(valid_depth)) * self.depth_scale if len(valid_depth) > 0 else 0,
                'max_depth': float(np.max(valid_depth)) * self.depth_scale if len(valid_depth) > 0 else 0,
                'mean_depth': float(np.mean(valid_depth)) * self.depth_scale if len(valid_depth) > 0 else 0,
                'valid_pixels': int(len(valid_depth))
            }
            
            self.frame_count += 1
            
            return {
                'frame_index': frame_index,
                'timestamp': timestamp,
                'depth_image_path': os.path.relpath(depth_path, self.base_folder),
                'color_image_path': os.path.relpath(color_path, self.base_folder),
                'depth_stats': depth_stats,
                'camera_intrinsics': {
                    'color': {
                        'width': self.color_intrinsics.width,
                        'height': self.color_intrinsics.height,
                        'fx': self.color_intrinsics.fx,
                        'fy': self.color_intrinsics.fy,
                        'ppx': self.color_intrinsics.ppx,
                        'ppy': self.color_intrinsics.ppy
                    },
                    'depth': {
                        'width': self.depth_intrinsics.width,
                        'height': self.depth_intrinsics.height,
                        'fx': self.depth_intrinsics.fx,
                        'fy': self.depth_intrinsics.fy,
                        'ppx': self.depth_intrinsics.ppx,
                        'ppy': self.depth_intrinsics.ppy
                    },
                    'depth_scale': self.depth_scale
                },
                # 新增：直接返回图像数据用于LeRobot格式
                'color_image_data': color_image.copy(),
                'depth_image_data': depth_image.copy()
            }
            
        except Exception as e:
            print(f"捕获帧数据失败: {e}")
            return None
    
    def start_recording(self, base_path):
        """开始录制"""
        if not self.initialize_camera():
            return False
            
        if not self.setup_recording_folders(base_path):
            return False
            
        self.is_recording = True
        self.frame_count = 0
        print("✓ 相机录制已开始")
        return True
    
    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("✓ 相机录制已停止")
            except Exception as e:
                print(f"停止相机时出错: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.stop_recording()
        self.pipeline = None
        self.config = None
        self.align = None
        print("✓ 相机资源已清理")
    
    def get_camera_info(self):
        """获取相机信息"""
        if not self.color_intrinsics or not self.depth_intrinsics:
            return None
            
        return {
            'camera_model': 'Intel RealSense D415',
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps,
            'depth_scale': self.depth_scale,
            'color_intrinsics': {
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height,
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy
            },
            'depth_intrinsics': {
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy
            }
        }


def test_camera():
    """测试相机功能"""
    camera = D415CameraRecorder()
    
    try:
        if camera.initialize_camera():
            print("相机测试成功！")
            info = camera.get_camera_info()
            print("相机信息:")
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            print("相机测试失败！")
    finally:
        camera.cleanup()


if __name__ == "__main__":
    test_camera()