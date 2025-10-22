"""
验证夹爪数据是否正确记录在LeRobot数据集中
"""
import pandas as pd
import sys
from pathlib import Path

def check_gripper_data(dataset_path):
    """检查数据集中的夹爪数据"""
    dataset_path = Path(dataset_path)
    
    # 读取数据
    parquet_file = dataset_path / "data" / "chunk-000" / "file-000000.parquet"
    
    if not parquet_file.exists():
        print(f"❌ 数据文件不存在: {parquet_file}")
        return False
    
    df = pd.read_parquet(parquet_file)
    
    print("="*60)
    print("夹爪数据验证")
    print("="*60)
    
    # 检查夹爪列是否存在
    if 'action.gripper' not in df.columns:
        print("❌ 未找到 action.gripper 列")
        return False
    
    print(f"\n✓ 找到 action.gripper 列")
    
    # 统计夹爪状态
    gripper_values = df['action.gripper'].unique()
    print(f"\n📊 夹爪状态值: {sorted(gripper_values)}")
    
    # 统计各状态的帧数
    print(f"\n📈 夹爪状态分布:")
    for value in sorted(gripper_values):
        count = (df['action.gripper'] == value).sum()
        percentage = count / len(df) * 100
        status_name = "闭合" if value == 1.0 else ("打开" if value == 0.0 else "未知")
        print(f"  {status_name} ({value}): {count}/{len(df)} ({percentage:.1f}%)")
    
    # 显示一些示例数据
    print(f"\n📝 前10帧的夹爪状态:")
    print(df[['frame_index', 'timestamp', 'action.gripper']].head(10))
    
    # 检查是否有状态变化
    changes = (df['action.gripper'].diff() != 0).sum()
    print(f"\n🔄 夹爪状态变化次数: {changes}")
    
    if changes > 0:
        print(f"✓ 检测到夹爪状态变化，数据有效")
    else:
        print(f"⚠️ 夹爪状态未变化（可能始终保持同一状态）")
    
    print("\n" + "="*60)
    print("✅ 夹爪数据验证完成")
    print("="*60)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # 查找最新的数据集
        import os
        datasets = [d for d in os.listdir('.') if d.startswith('lerobot_dataset_')]
        if not datasets:
            print("❌ 未找到数据集")
            print("用法: python check_gripper_data.py <数据集路径>")
            sys.exit(1)
        
        # 按时间排序，选择最新的
        datasets.sort(reverse=True)
        dataset_path = datasets[0]
        print(f"使用最新数据集: {dataset_path}\n")
    
    check_gripper_data(dataset_path)
