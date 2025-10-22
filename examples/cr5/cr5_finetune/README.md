# CR5 Pi0 微调

##  训练命令

```bash
cd /home/huang/learn_arm_robot/openpi

# LoRA 微调 (推荐, RTX 4090 24GB)
./examples/cr5/cr5_finetune/train_cr5.sh pi0_cr5_finetune_lora

# 完全微调 (需要 >70GB 显存)
./examples/cr5/cr5_finetune/train_cr5.sh pi0_cr5_finetune

# 降低显存占用 (RTX 4070 Ti 12GB 可尝试)
./examples/cr5/cr5_finetune/train_cr5.sh pi0_cr5_finetune_lora --batch-size 8
```

## 📊 查看训练

```bash
# 查看日志
tail -f checkpoints/pi0_cr5_finetune_lora/*/train.log

# WandB 面板会自动打开
```

## 🔧 部署推理

```bash
python scripts/serve_policy.py \
  --checkpoint-dir checkpoints/pi0_cr5_finetune_lora/cr5_lora_YYYYMMDD_HHMMSS \
  --host 0.0.0.0 \
  --port 5000
```

---

详细文档: `CR5_README.md` | `WORKSPACE_SUMMARY.md`
