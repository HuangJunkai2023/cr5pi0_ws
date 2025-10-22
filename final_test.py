#!/usr/bin/env python3
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 70)
print("  CR5 Pi0 微调环境最终测试")
print("=" * 70)
print()

print("测试 1/2: 配置加载...")
try:
    sys.path.insert(0, 'src')
    from openpi.training import config
    
    configs = ["pi0_cr5_finetune", "pi0_fast_cr5_finetune", "pi0_cr5_finetune_lora"]
    for name in configs:
        cfg = config.get_config(name)
        assert cfg.model.action_dim == 7
    print("  ✅ 所有配置加载成功")
    print()
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

print("测试 2/2: 数据集加载...")
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("cr5_test_dataset", tolerance_s=0.4)
    print(f"  ✅ 数据集OK - 轨迹:{dataset.num_episodes}, 帧:{dataset.num_frames}")
    print()
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

print("=" * 70)
print("  ✅ 所有测试通过！")
print("=" * 70)
