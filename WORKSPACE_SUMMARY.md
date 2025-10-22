# CR5 工作空间整理总结

## ✅ 已完成的工作

### 1. 数据采集改进
- **文件**: `examples/cr5/CR5_TCP_test/lerobot_collect.py`
- **改进**: 精确 10Hz 控制循环,避免时间戳漂移
- **特性**: 
  - 固定时间基准 (累积式定时)
  - 实时 FPS 监控 (颜色编码)
  - 异步夹爪控制 (避免阻塞)

### 2. 数据修复工具
- **文件**: `examples/cr5/CR5_TCP_test/fix_timestamps.py`
- **功能**: 重采样时间戳为严格 10Hz
- **使用**: `python fix_timestamps.py` (自动备份)

### 3. 策略配置 (基于 UR5 模式)
- **文件**: `src/openpi/policies/cr5_policy.py`
- **关键改动**:
  ```python
  # CR5Inputs: model_type = PI0 (不是 PI0_FAST)
  # CR5Outputs: 只取前 7 维动作
  return {"actions": data["actions"][:, :7]}
  ```

### 4. 训练配置
- **文件**: `src/openpi/training/config.py`
- **配置**:
  - `pi0_cr5_finetune`: 完全微调 (>70GB)
  - `pi0_fast_cr5_finetune`: Fast 版本 (>70GB)
  - `pi0_cr5_finetune_lora`: LoRA 微调 (>22GB) ⭐推荐
- **改动**: 
  - 不指定 `action_dim` (使用默认 32)
  - 复用 UR5 归一化统计 (`asset_id="ur5e"`)
  - 可加载完整 Pi0 Base 预训练权重

### 5. 数据加载修复
- **文件**: `src/openpi/training/data_loader.py`
- **修改**: 
  ```python
  # Line 133-141
  tolerance_s = 0.5 if "cr5" in repo_id.lower() else 0.0001
  ```
- **原因**: 真实机器人时序抖动

### 6. 一键训练脚本
- **文件**: `train_cr5.sh`
- **用法**: `./train_cr5.sh pi0_cr5_finetune_lora`
- **功能**: 自动运行 compute_norm_stats + train

## 🗑️ 已删除的文件

- `CR5_README.md` (旧版)
- `QUICK_START_CR5.txt` (旧版)
- `CR5_TRAINING_GUIDE.md` (旧版)
- `CR5_SETUP_SUMMARY.md` (旧版)
- `examples/cr5/CR5_TCP_test/check_cr5_env.sh` (测试脚本)
- `examples/cr5/CR5_TCP_test/final_test.py` (测试脚本)

## 📝 新增文件

- **`CR5_README.md`**: 精简版完整指南 (包含所有关键信息)

## 🎯 当前状态

### 数据准备
- ✅ 数据集: 5 episodes, 530 frames, 10 FPS
- ✅ 时间戳修复: 已重采样为严格 10Hz
- ✅ 格式: LeRobot 标准格式
- ✅ 位置: `~/.cache/huggingface/lerobot/cr5_test_dataset`

### 训练准备
- ✅ 策略: CR5Inputs/Outputs (基于 UR5 模式)
- ✅ 配置: 3 个训练配置可用
- ✅ 预训练权重: Pi0 Base 已下载 (~12GB)
- ⚠️ 显存: RTX 4070 Ti 12GB **不足**

### 下一步
1. **等待 RTX 4090** - 需要 24GB 显存
2. **或降低 batch size** - `--batch-size 8` (但会变慢)
3. **收集更多数据** - 建议 20-50 episodes

## 🔧 技术要点

### 核心创新: UR5 模式适配

**问题**: CR5 7 维动作 vs Pi0 32 维预训练

**解决**: 
1. 训练时填充到 32 维 (PadStatesAndActions)
2. 推理时只取前 7 维 (CR5Outputs)
3. 复用 UR5 归一化统计 (同为 7 维单臂)

### 关键修改位置

```
src/openpi/
├── policies/cr5_policy.py
│   ├── CR5Inputs: model_type=PI0 ✅
│   └── CR5Outputs: [:, :7] ✅
└── training/
    ├── config.py
    │   └── 3 configs: action_dim=32 (默认) ✅
    └── data_loader.py
        └── tolerance_s=0.5 for CR5 ✅
```

## 📊 配置对比

| 配置 | 显存 | 速度 | 效果 | 状态 |
|------|------|------|------|------|
| pi0_cr5_finetune | >70GB | 慢 | 最好 | ❌ 4070Ti 不够 |
| pi0_fast_cr5_finetune | >70GB | 快 | 很好 | ❌ 4070Ti 不够 |
| pi0_cr5_finetune_lora | >22GB | 中 | 好 | ⚠️ 4070Ti 边缘 |

## 🚀 快速命令

```bash
# 数据采集
cd examples/cr5/CR5_TCP_test && python lerobot_collect.py

# 修复时间戳
python fix_timestamps.py

# 训练 (需要 4090)
./train_cr5.sh pi0_cr5_finetune_lora

# 查看日志
tail -f checkpoints/pi0_cr5_finetune_lora/*/train.log
```

---

**整理时间**: 2025-10-22 13:50
**工作空间**: 已清理,保留核心文件
**状态**: ✅ 配置完成,等待硬件升级
