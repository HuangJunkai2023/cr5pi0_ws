# CR5 Pi0 微调
##  训练命令
```bash
cd /home/huang/learn_arm_robot/openpi
# LoRA 微调 (推荐, RTX 4090 24GB)
./examples/cr5/cr5_finetune/train_cr5.sh pi0_cr5_finetune_lora
# 完全微调 (需要 >70GB 显存)
./examples/cr5/cr5_finetune/train_cr5.sh pi0_cr5_finetune
```
## 📊 查看训练
```bash
# 查看日志
tail -f checkpoints/pi0_cr5_finetune_lora/*/train.log
# WandB 面板会自动打开
```
## 启动推理服务器
```bash
uv run scripts/serve_policy.py policy:checkpoint     --policy.config=pi0_cr5_finetune_lora    --policy.dir=checkpoints/pi0_cr5_finetune_lora/cr5_test_dataset/19999
```
## 启动cr5客户端
```bash
uv run examples/cr5/cr5_client.py --host 127.0.1.1 --port 8000 --robot_ip 192.168.5.1
```