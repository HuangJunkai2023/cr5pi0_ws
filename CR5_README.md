# CR5 机器人 Pi0 训练指南

本指南介绍如何使用 OpenPI 框架训练 CR5 机械臂执行抓取任务。

## 🤖 CR5 机器人规格

- **自由度**: 6 关节 + 1 夹爪 = 7 维动作空间
- **观察**: 1 个顶部相机 (640×480 RGB)
- **状态**: 6 个关节位置 (弧度)
- **动作**: 6 个关节速度 + 1 个夹爪位置 [0.0-1.0]
- **控制频率**: 10 Hz

## 📁 项目结构

```
openpi/
├── src/openpi/
│   ├── policies/cr5_policy.py          # CR5 数据转换 (基于 UR5 模式)
│   └── training/
│       ├── config.py                   # 训练配置 (3 个 CR5 配置)
│       └── data_loader.py              # 数据加载 (tolerance_s=0.5)
├── examples/cr5/CR5_TCP_test/
│   ├── lerobot_collect.py              # 数据采集脚本 (改进的 10Hz 控制)
│   └── fix_timestamps.py               # 时间戳修复工具
├── train_cr5.sh                        # 一键训练脚本
└── checkpoints/pi0_cr5_finetune_lora/  # 训练输出 (checkpoint)
```

## 🚀 快速开始

### 1. 数据采集

```bash
cd examples/cr5/CR5_TCP_test
python lerobot_collect.py
```

**关键参数**:
- `FPS = 10` - 采集频率 10 Hz (精确控制循环)
- `ROBOT_IP = "192.168.5.1"` - CR5 机器人 IP
- `TASK_NAME = "grasp_cube"` - 任务名称

**数据格式** (LeRobot):
- `observation.state`: [6] 关节位置
- `observation.images.top`: (480, 640, 3) RGB 图像
- `action`: [7] (6 关节速度 + 1 夹爪位置)

### 2. 修复时间戳 (如果需要)

```bash
cd examples/cr5/CR5_TCP_test
python fix_timestamps.py
```

自动将不规则时间戳重采样为严格 10 Hz。

### 3. 训练模型

```bash
./train_cr5.sh pi0_cr5_finetune_lora
```

**可用配置**:
| 配置名 | 模型 | 显存需求 | 说明 |
|--------|------|---------|------|
| `pi0_cr5_finetune_lora` | Pi0 LoRA | >22 GB | **推荐** (4090 可用) |
| `pi0_cr5_finetune` | Pi0 全量 | >70 GB | 完全微调 |
| `pi0_fast_cr5_finetune` | Pi0-FAST | >70 GB | 快速推理版本 |

## 🔧 关键技术细节

### 1. 动作空间适配 (基于 UR5 模式)

**原理**: Pi0 Base 预训练使用 32 维动作，CR5 只有 7 维。

```python
# 训练时: CR5 7 维 → 填充到 32 维
# 推理时: 模型输出 32 维 → 只取前 7 维

class CR5Outputs:
    def __call__(self, data):
        return {"actions": data["actions"][:, :7]}  # 只取前 7 维
```

### 2. 时间戳容忍度

```python
# src/openpi/training/data_loader.py (line 133-141)
tolerance_s = 0.5 if "cr5" in repo_id.lower() else 0.0001
```

**原因**: 真实机器人采集存在时序抖动 (0.09-0.12s 间隔)。

### 3. 归一化统计复用

```python
# 复用 UR5 的归一化统计 (同为 7 维单臂机器人)
assets=AssetsConfig(
    assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
    asset_id="ur5e",
)
```

## 📊 训练监控

```bash
# 查看训练日志
tail -f checkpoints/pi0_cr5_finetune_lora/*/train.log

# WandB 离线模式
wandb offline  # 已默认设置
```

**训练参数**:
- Steps: 20,000
- Batch Size: 32 (可用 `--batch-size 8` 降低显存)
- 保存间隔: 每 1000 步
- 保留检查点: 每 5000 步

## ⚠️ 常见问题

### 1. 显存不足 (OOM)

**问题**: RTX 4070 Ti 12GB 可能不够。

**解决**:
```bash
# 降低 batch size
./train_cr5.sh pi0_cr5_finetune_lora --batch-size 8

# 或等待使用 RTX 4090 24GB
```

### 2. 时间戳越界错误

**问题**: `Invalid frame index=107 for numFrames=107`

**解决**: 运行时间戳修复工具
```bash
python examples/cr5/CR5_TCP_test/fix_timestamps.py
```

### 3. 采集频率不稳定

**问题**: 实际 FPS < 10 Hz

**改进**: 已在 `lerobot_collect.py` 中实现精确定时控制:
```python
next_frame_time += target_period  # 累积式时间基准
sleep(next_frame_time - current_time)  # 精确睡眠
```

## 📚 参考资料

- **OpenPI 官方**: [github.com/physical-intelligence/openpi](https://github.com/physical-intelligence/openpi)
- **UR5 示例**: `examples/ur5/README.md` (CR5 基于此实现)
- **LeRobot 格式**: [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot)
- **博客教程**: [Pi0 微调流程](https://blog.csdn.net/nenchoumi3119/article/details/148688800)

## 🎯 下一步

1. **收集更多数据** - 5 episodes 可能不够，建议 20-50 episodes
2. **调整任务提示** - 修改 `default_prompt="grasp_cube"` 适配你的任务
3. **测试推理** - 训练后使用 `serve_policy.py` 部署模型
4. **优化性能** - 根据成功率调整超参数

---

**最后更新**: 2025-10-22  
**状态**: ✅ 配置完成，基于 UR5 模式，可以加载 Pi0 Base 预训练权重
