"""
合并多个LeRobot数据集为单个包含多个episodes的数据集
用法: python merge_datasets.py
"""

import pandas as pd
import json
import shutil
from pathlib import Path
from datetime import datetime
import cv2


def merge_lerobot_datasets(dataset_folders, output_name=None):
    """
    合并多个LeRobot数据集文件夹为一个包含多个episodes的数据集
    
    Args:
        dataset_folders: 要合并的数据集文件夹路径列表
        output_name: 输出数据集名称，如果为None则自动生成
    """
    
    if not dataset_folders:
        print("❌ 没有提供数据集文件夹")
        return None
    
    # 创建输出目录
    if output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"lerobot_dataset_merged_{timestamp}"
    
    output_path = Path(output_name)
    if output_path.exists():
        print(f"⚠️  输出目录已存在: {output_path}")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("取消合并")
            return None
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True)
    (output_path / "meta").mkdir(parents=True)
    (output_path / "videos" / "observation.images.camera_color").mkdir(parents=True)
    (output_path / "videos" / "observation.images.camera_depth").mkdir(parents=True)
    
    print(f"\n📦 开始合并 {len(dataset_folders)} 个数据集...")
    print(f"📁 输出目录: {output_path}")
    
    # 收集所有数据
    all_frames = []
    all_episodes = []
    all_tasks = set()
    
    current_episode_index = 0
    total_frames = 0
    
    for folder_idx, folder in enumerate(dataset_folders):
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"⚠️  跳过不存在的文件夹: {folder}")
            continue
        
        print(f"\n处理 [{folder_idx + 1}/{len(dataset_folders)}]: {folder_path.name}")
        
        try:
            # 读取帧数据
            data_file = folder_path / "data" / "chunk-000" / "file-000000.parquet"
            if not data_file.exists():
                print(f"  ⚠️  未找到数据文件: {data_file}")
                continue
            
            df = pd.read_parquet(data_file)
            print(f"  ✓ 读取 {len(df)} 帧数据")
            
            # 读取episode元数据
            episodes_file = folder_path / "meta" / "episodes.parquet"
            if episodes_file.exists():
                episodes_df = pd.read_parquet(episodes_file)
                
                # 处理每个episode
                for _, episode in episodes_df.iterrows():
                    old_ep_idx = int(episode['episode_index'])
                    
                    # 获取该episode的所有帧
                    episode_frames = df[df['episode_index'] == old_ep_idx].copy()
                    
                    if len(episode_frames) == 0:
                        continue
                    
                    # 更新episode索引
                    episode_frames['episode_index'] = current_episode_index
                    
                    # 更新全局索引
                    episode_frames['index'] = range(total_frames, total_frames + len(episode_frames))
                    
                    # 重置frame索引
                    episode_frames['frame_index'] = range(len(episode_frames))
                    
                    # 更新视频引用中的episode编号
                    for img_col in ['observation.images.camera_color', 'observation.images.camera_depth']:
                        if img_col in episode_frames.columns:
                            def update_video_path(val):
                                if isinstance(val, str):
                                    video_info = json.loads(val)
                                    old_path = video_info['path']
                                    # 更新路径中的episode编号
                                    new_path = old_path.replace(
                                        f"episode_{old_ep_idx:06d}",
                                        f"episode_{current_episode_index:06d}"
                                    )
                                    video_info['path'] = new_path
                                    return json.dumps(video_info)
                                return val
                            
                            episode_frames[img_col] = episode_frames[img_col].apply(update_video_path)
                    
                    # 复制视频文件
                    for video_type in ['observation.images.camera_color', 'observation.images.camera_depth']:
                        video_type_short = video_type.split('.')[-1]
                        old_video = folder_path / "videos" / video_type / f"episode_{old_ep_idx:06d}.mp4"
                        new_video = output_path / "videos" / video_type / f"episode_{current_episode_index:06d}.mp4"
                        
                        if old_video.exists():
                            shutil.copy2(old_video, new_video)
                            print(f"  ✓ 复制视频: episode_{current_episode_index:06d}.mp4 ({video_type_short})")
                    
                    # 添加到总数据
                    all_frames.append(episode_frames)
                    
                    # 创建新的episode元数据
                    new_episode = {
                        'episode_index': current_episode_index,
                        'length': len(episode_frames),
                        'timestamp': datetime.now().isoformat(),
                        'tasks': episode.get('tasks', ['teach_drag']),
                        'data/chunk_index': 0,
                        'data/file_index': 0,
                        f'videos/{video_type}/chunk_index': 0,
                        f'videos/{video_type}/file_index': current_episode_index,
                        f'videos/{video_type}/from_timestamp': 0.0,
                        f'videos/{video_type}/to_timestamp': float((len(episode_frames) - 1) / 10),
                    }
                    
                    # 添加两个视频类型的元数据
                    for video_type in ['observation.images.camera_color', 'observation.images.camera_depth']:
                        new_episode[f'videos/{video_type}/chunk_index'] = 0
                        new_episode[f'videos/{video_type}/file_index'] = current_episode_index
                        new_episode[f'videos/{video_type}/from_timestamp'] = 0.0
                        new_episode[f'videos/{video_type}/to_timestamp'] = float((len(episode_frames) - 1) / 10)
                    
                    all_episodes.append(new_episode)
                    
                    # 收集任务
                    tasks = episode.get('tasks', ['teach_drag'])
                    if isinstance(tasks, list):
                        all_tasks.update(tasks)
                    
                    total_frames += len(episode_frames)
                    current_episode_index += 1
                    
                    print(f"  ✓ Episode {current_episode_index - 1}: {len(episode_frames)} 帧")
            
        except Exception as e:
            print(f"  ❌ 处理出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_frames:
        print("\n❌ 没有成功处理任何数据")
        return None
    
    print(f"\n📊 合并统计:")
    print(f"  总Episodes: {len(all_episodes)}")
    print(f"  总帧数: {total_frames}")
    print(f"  任务类型: {list(all_tasks)}")
    
    # 合并所有帧数据
    print(f"\n💾 保存合并后的数据...")
    merged_df = pd.concat(all_frames, ignore_index=True)
    
    # 保存为parquet
    output_data_file = output_path / "data" / "chunk-000" / "file-000000.parquet"
    merged_df.to_parquet(output_data_file, index=False)
    print(f"  ✓ 帧数据已保存: {output_data_file}")
    
    # 保存episodes元数据
    episodes_df = pd.DataFrame(all_episodes)
    episodes_output = output_path / "meta" / "episodes.parquet"
    episodes_df.to_parquet(episodes_output, index=False)
    print(f"  ✓ Episodes元数据已保存: {episodes_output}")
    
    # 读取第一个数据集的info.json作为模板
    first_folder = Path(dataset_folders[0])
    info_template = {}
    if (first_folder / "meta" / "info.json").exists():
        with open(first_folder / "meta" / "info.json", 'r', encoding='utf-8') as f:
            info_template = json.load(f)
    
    # 更新info.json
    info_template['total_episodes'] = len(all_episodes)
    info_template['total_frames'] = total_frames
    
    info_output = output_path / "meta" / "info.json"
    with open(info_output, 'w', encoding='utf-8') as f:
        json.dump(info_template, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Info已保存: {info_output}")
    
    # 保存tasks
    tasks_data = [{"task": task, "task_index": idx} for idx, task in enumerate(sorted(all_tasks))]
    tasks_df = pd.DataFrame(tasks_data)
    tasks_output = output_path / "meta" / "tasks.parquet"
    tasks_df.to_parquet(tasks_output, index=False)
    print(f"  ✓ Tasks已保存: {tasks_output}")
    
    # 计算统计信息
    print(f"\n📈 计算统计信息...")
    stats = {}
    
    # 数值特征统计
    numeric_features = [
        'observation.state.J1', 'observation.state.J2', 'observation.state.J3',
        'observation.state.J4', 'observation.state.J5', 'observation.state.J6',
        'observation.cartesian_position.X', 'observation.cartesian_position.Y',
        'observation.cartesian_position.Z', 'observation.cartesian_position.Rx',
        'observation.cartesian_position.Ry', 'observation.cartesian_position.Rz',
        'action.J1_vel', 'action.J2_vel', 'action.J3_vel',
        'action.J4_vel', 'action.J5_vel', 'action.J6_vel', 'action.gripper'
    ]
    
    for feature in numeric_features:
        if feature in merged_df.columns:
            stats[feature] = {
                "min": float(merged_df[feature].min()),
                "max": float(merged_df[feature].max()),
                "mean": float(merged_df[feature].mean()),
                "std": float(merged_df[feature].std())
            }
    
    stats_output = output_path / "meta" / "stats.json"
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 统计信息已保存: {stats_output}")
    
    # 创建README
    readme_content = f"""---
license: apache-2.0
tags:
- LeRobot
- robotics
- dobot
- teach-drag
task_categories:
- robotics
---

# {output_name}

合并的LeRobot格式机器人数据集，包含多个示教轨迹。

## 数据集信息

- **机器人类型**: dobot_cr5
- **总Episodes**: {len(all_episodes)}
- **总帧数**: {total_frames}
- **任务类型**: {', '.join(sorted(all_tasks))}
- **创建时间**: {datetime.now().isoformat()}

## 使用方法

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("{output_name}")

# 获取一帧数据
frame = dataset[0]
print(frame.keys())
```

## 数据来源

该数据集由以下 {len(dataset_folders)} 个数据集合并而成：

"""
    
    for idx, folder in enumerate(dataset_folders):
        readme_content += f"{idx + 1}. {Path(folder).name}\n"
    
    readme_content += f"""
合并时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_output = output_path / "README.md"
    with open(readme_output, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ README已创建: {readme_output}")
    
    print("\n" + "="*60)
    print("✅ 数据集合并完成！")
    print("="*60)
    print(f"📁 输出路径: {output_path.absolute()}")
    print(f"📊 Episodes: {len(all_episodes)}")
    print(f"📊 总帧数: {total_frames}")
    print("="*60)
    
    return str(output_path.absolute())


def auto_find_datasets(base_dir="."):
    """自动查找目录下的所有LeRobot数据集"""
    base_path = Path(base_dir)
    datasets = []
    
    for item in sorted(base_path.glob("lerobot_dataset_*")):
        if item.is_dir() and not item.name.endswith("_merged"):
            # 检查是否包含必要的文件
            if (item / "data" / "chunk-000" / "file-000000.parquet").exists():
                datasets.append(str(item))
    
    return datasets


if __name__ == "__main__":
    print("="*60)
    print("LeRobot 数据集合并工具")
    print("="*60)
    
    # 自动查找数据集
    print("\n🔍 搜索当前目录下的数据集...")
    datasets = auto_find_datasets(".")
    
    if not datasets:
        print("❌ 未找到任何数据集文件夹")
        print("提示: 数据集文件夹应该以 'lerobot_dataset_' 开头")
        exit(1)
    
    print(f"\n找到 {len(datasets)} 个数据集:")
    for idx, dataset in enumerate(datasets):
        folder_name = Path(dataset).name
        # 读取基本信息
        try:
            df = pd.read_parquet(Path(dataset) / "data" / "chunk-000" / "file-000000.parquet")
            num_frames = len(df)
            num_episodes = df['episode_index'].nunique()
            print(f"  [{idx + 1}] {folder_name}: {num_episodes} episode(s), {num_frames} 帧")
        except:
            print(f"  [{idx + 1}] {folder_name}")
    
    # 询问是否合并所有数据集
    print("\n" + "="*60)
    response = input(f"是否合并所有 {len(datasets)} 个数据集? (y/n): ")
    
    if response.lower() != 'y':
        print("取消合并")
        exit(0)
    
    # 执行合并
    output = merge_lerobot_datasets(datasets)
    
    if output:
        print(f"\n✅ 成功！合并后的数据集可用于 pi0 训练")
        print(f"📂 数据集路径: {output}")
