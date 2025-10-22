#!/bin/bash
# CR5 Pi0 微调训练脚本
#
# 用法:
#   ./train_cr5.sh pi0_cr5_finetune          # 完全微调 (需要 >70GB VRAM)
#   ./train_cr5.sh pi0_fast_cr5_finetune     # FAST 微调 (需要 >70GB VRAM)
#   ./train_cr5.sh pi0_cr5_finetune_lora     # LoRA 微调 (需要 >22.5GB VRAM, 推荐)

set -e

CONFIG_NAME="${1:-pi0_cr5_finetune_lora}"
EXP_NAME="${2:-cr5_test_dataset}"

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================="
echo "  CR5 Pi0 微调训练"
echo "============================================="
echo "项目目录: $PROJECT_ROOT"
echo "配置: $CONFIG_NAME"
echo "实验名: $EXP_NAME"
echo "============================================="
echo ""

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 创建数据集软链接 (如果不存在)
LEROBOT_CACHE="$HOME/.cache/huggingface/lerobot"
DATASET_LINK="$LEROBOT_CACHE/$EXP_NAME"
DATASET_SOURCE="/home/huang/learn_arm_robot/openpi/examples/cr5/CR5_TCP_test/lerobot_data"

if [ ! -L "$DATASET_LINK" ] && [ ! -d "$DATASET_LINK" ]; then
    echo "创建数据集软链接..."
    mkdir -p "$LEROBOT_CACHE"
    ln -s "$DATASET_SOURCE" "$DATASET_LINK"
    echo "✓ 数据集链接创建成功: $DATASET_LINK -> $DATASET_SOURCE"
else
    echo "✓ 数据集已存在: $DATASET_LINK"
fi
echo ""

# 步骤 1: 计算归一化统计
echo "步骤 1/2: 计算归一化统计信息..."
echo "这一步可能需要几分钟时间，请耐心等待..."
echo ""
uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 归一化统计计算失败！"
    echo "请检查:"
    echo "  1. 数据集路径是否正确"
    echo "  2. 数据格式是否正确"
    echo "  3. 是否有足够的内存"
    exit 1
fi

echo ""
echo "✓ 归一化统计计算完成！"
echo ""

# 步骤 2: 开始训练
echo "步骤 2/2: 开始训练..."
echo ""
uv run scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 训练失败！"
    echo "请检查错误信息"
    exit 1
fi

echo ""
echo "============================================="
echo "  ✓ 训练完成！"
echo "============================================="
echo "检查点保存在: ./checkpoints/$CONFIG_NAME/$EXP_NAME/"
echo ""
