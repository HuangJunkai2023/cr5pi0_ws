#!/usr/bin/env python3
"""
修复 LeRobot 数据集的时间戳,使其严格对齐到指定 FPS

用法:
    python fix_timestamps.py [数据集路径]
"""

import sys
import json
import shutil
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from datetime import datetime


def fix_episode_timestamps(episode_file: Path, target_fps: float = 10.0):
    """
    修复单个 episode 的时间戳,使其严格按照目标 FPS 排列
    
    Args:
        episode_file: episode parquet 文件路径
        target_fps: 目标帧率
    """
    # 读取数据
    table = pq.read_table(episode_file)
    df = table.to_pandas()
    
    num_frames = len(df)
    
    # 生成严格的时间戳: 0.0, 0.1, 0.2, ...
    target_period = 1.0 / target_fps
    new_timestamps = np.arange(num_frames, dtype=np.float64) * target_period
    
    # 获取原始时间戳统计
    old_timestamps = df['timestamp'].values
    old_duration = old_timestamps[-1] - old_timestamps[0]
    new_duration = new_timestamps[-1] - new_timestamps[0]
    old_fps = (num_frames - 1) / old_duration if old_duration > 0 else 0
    
    print(f"  原始: {num_frames} 帧, {old_duration:.3f}s, 平均 {old_fps:.2f} Hz")
    print(f"  修复: {num_frames} 帧, {new_duration:.3f}s, 严格 {target_fps:.2f} Hz")
    
    # 更新时间戳
    df['timestamp'] = new_timestamps
    
    # 转回 Arrow Table 并保存
    new_table = pa.Table.from_pandas(df, schema=table.schema)
    pq.write_table(new_table, episode_file)
    
    return old_fps, target_fps


def fix_dataset(dataset_path: Path, target_fps: float = 10.0, create_backup: bool = True):
    """
    修复整个数据集的时间戳
    
    Args:
        dataset_path: 数据集根目录
        target_fps: 目标帧率
        create_backup: 是否创建备份
    """
    print("="*60)
    print("🔧 LeRobot 时间戳修复工具")
    print("="*60)
    print(f"数据集: {dataset_path}")
    print(f"目标 FPS: {target_fps}")
    print()
    
    # 验证路径
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        return False
    
    meta_dir = dataset_path / 'meta'
    data_dir = dataset_path / 'data'
    
    if not meta_dir.exists() or not data_dir.exists():
        print(f"❌ 错误: 不是有效的 LeRobot 数据集 (缺少 meta/ 或 data/)")
        return False
    
    # 创建备份
    if create_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = dataset_path.parent / f"{dataset_path.name}_backup_{timestamp}"
        
        print(f"📦 创建备份: {backup_dir.name}")
        shutil.copytree(dataset_path, backup_dir)
        print(f"✅ 备份完成")
        print()
    
    # 读取 episodes 信息
    episodes_file = meta_dir / 'episodes.jsonl'
    episodes = []
    with open(episodes_file) as f:
        for line in f:
            episodes.append(json.loads(line))
    
    print(f"📊 找到 {len(episodes)} 个 episodes")
    print()
    
    # 处理每个 episode
    for ep in episodes:
        ep_idx = ep['episode_index']
        chunk_idx = ep.get('chunk_index', 0)
        
        # 查找 episode 数据文件
        episode_file = data_dir / f'chunk-{chunk_idx:03d}' / f'episode_{ep_idx:06d}.parquet'
        
        if not episode_file.exists():
            print(f"⚠️  警告: 找不到 episode {ep_idx} 的数据文件")
            continue
        
        print(f"🔧 修复 Episode {ep_idx}:")
        old_fps, new_fps = fix_episode_timestamps(episode_file, target_fps)
        print()
    
    # 更新 info.json 中的 FPS (如果需要)
    info_file = meta_dir / 'info.json'
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
        
        old_info_fps = info.get('fps')
        if old_info_fps != target_fps:
            print(f"📝 更新 info.json: fps {old_info_fps} → {target_fps}")
            info['fps'] = target_fps
            
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
    
    print("="*60)
    print("✅ 时间戳修复完成!")
    print("="*60)
    print()
    print("接下来:")
    print("1. 验证修复结果 (可选)")
    print("2. 重新运行训练脚本")
    print()
    
    return True


def main():
    if len(sys.argv) < 2:
        # 默认使用缓存中的数据集
        dataset_path = Path.home() / '.cache/huggingface/lerobot/cr5_test_dataset'
        print(f"使用默认数据集: {dataset_path}")
    else:
        dataset_path = Path(sys.argv[1])
    
    # 执行修复
    success = fix_dataset(dataset_path, target_fps=10.0, create_backup=True)
    
    if success:
        print("💡 提示: 如果还遇到越界错误,可能需要重新编码视频")
        print("   参考方案 2: 重新编码视频文件")
    else:
        print("❌ 修复失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
