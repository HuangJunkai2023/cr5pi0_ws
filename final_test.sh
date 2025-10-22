#!/usr/bin/env python3#!/bin/bash

"""CR5 Pi0 微调最终测试脚本"""

echo "=========================================="

import osecho "CR5 训练流程最终测试"

import sysecho "=========================================="

echo ""

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置环境变量

print("=" * 70)export HF_ENDPOINT="https://hf-mirror.com"

print("  CR5 Pi0 微调环境最终测试")

print("=" * 70)echo "1. 测试配置加载..."

print()uv run python test_cr5_config.py

if [ $? -ne 0 ]; then

# Test 1: Configuration loading    echo "❌ 配置测试失败"

print("测试 1/2: 配置加载...")    exit 1

try:fi

    sys.path.insert(0, 'src')

    from openpi.training import configecho ""

    echo "2. 测试数据加载..."

    configs = ["pi0_cr5_finetune", "pi0_fast_cr5_finetune", "pi0_cr5_finetune_lora"]uv run python -c "

    for name in configs:from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        cfg = config.get_config(name)ds = LeRobotDataset('cr5_test_dataset', tolerance_s=0.4)

        assert cfg.model.action_dim == 7print(f'✅ 数据集加载成功')

        assert cfg.model.action_horizon == 10print(f'   - 总轨迹: {ds.num_episodes}')

        assert cfg.batch_size == 32print(f'   - 总帧数: {ds.num_frames}')

    print("  ✅ 所有配置加载成功 (3/3)")print(f'   - FPS: {ds.fps}')

    print()"

except Exception as e:

    print(f"  ❌ 配置加载失败: {e}")if [ $? -ne 0 ]; then

    import traceback    echo "❌ 数据加载测试失败"

    traceback.print_exc()    exit 1

    sys.exit(1)fi



# Test 2: Dataset loadingecho ""

print("测试 2/2: 数据集加载...")echo "=========================================="

try:echo "✅ 所有测试通过！"

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetecho "=========================================="

    echo ""

    dataset = LeRobotDataset("cr5_test_dataset", tolerance_s=0.4)echo "现在可以开始训练了:"

    print(f"  ✅ 数据集加载成功！")echo ""

    print(f"     - 总轨迹: {dataset.num_episodes}")echo "方式1 - 使用便捷脚本:"

    print(f"     - 总帧数: {dataset.num_frames}")echo "  ./train_cr5.sh pi0_cr5_finetune_lora"

    print(f"     - FPS: {dataset.fps}")echo ""

    print()echo "方式2 - 手动执行:"

except Exception as e:echo "  export HF_ENDPOINT=\"https://hf-mirror.com\""

    print(f"  ❌ 数据集加载失败: {e}")echo "  uv run scripts/compute_norm_stats.py --config-name pi0_cr5_finetune_lora"

    import tracebackecho "  uv run scripts/train.py pi0_cr5_finetune_lora --exp-name=cr5_test_dataset --overwrite"

    traceback.print_exc()echo ""

    sys.exit(1)echo "注意: 在 RTX 4090 上训练时，建议修改 batch_size 为 16 或 32"

echo "=========================================="

# Summary
print("=" * 70)
print("  ✅ 所有测试通过！")
print("=" * 70)
print()
print("下一步:")
print("  1. 计算归一化统计:")
print("     export HF_ENDPOINT='https://hf-mirror.com'")
print("     uv run scripts/compute_norm_stats.py --config-name pi0_cr5_finetune_lora")
print()
print("  2. 开始训练:")
print("     uv run scripts/train.py pi0_cr5_finetune_lora --exp-name=cr5_test_dataset --overwrite")
print()
print("推荐配置:")
print("  - pi0_cr5_finetune_lora: LoRA 微调 (>22.5GB VRAM, 适合 RTX 4090)")
print("  - pi0_fast_cr5_finetune: FAST 完全微调 (>70GB VRAM, 需要 A100)")
print("  - pi0_cr5_finetune: Pi0 完全微调 (>70GB VRAM, 需要 A100)")
print()
