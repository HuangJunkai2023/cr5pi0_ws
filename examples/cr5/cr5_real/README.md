# CR5 拖拽示教 - pi0 数据采集

## 🎯 一键采集 pi0 训练数据

只需运行 `simple_drag_demo.py`，就可以通过拖拽示教自动采集 pi0 格式的训练数据！

## 🚀 快速开始

```bash
cd /home/huang/learn_arm_robot/openpi/examples/cr5/CR5_TCP_test

# 运行拖拽示教
python simple_drag_demo.py
```

就这么简单！程序会：
1. ✅ 自动连接机器人和相机
2. ✅ 开启拖拽模式
3. ✅ 实时记录关节角度和相机图像（10 Hz）
4. ✅ 自动转换角度→弧度
5. ✅ 自动计算关节速度
6. ✅ 保存为 pi0 训练格式

## 📝 使用流程

### 1. 启动程序

```bash
python simple_drag_demo.py
```

### 2. 等待初始化

程序会自动：
- 连接机器人（192.168.5.1）
- 初始化相机
- 使能机器人
- 开启拖拽模式

看到 `✓ 拖拽模式已开启!` 后即可开始示教。

### 3. 拖拽示教

- 手动拖动机器人完成任务动作
- 程序自动记录数据（10 Hz）
- 屏幕会显示当前关节角度和位置

示例输出：
```
记录点   1: 关节[ 36.6°,  4.7°,  1.1°,  0.4°, -4.6°,  0.0°] 位置[ 145.0,-378.4,1143.3] | 相机: ✓
记录点   2: 关节[ 36.8°,  4.8°,  1.2°,  0.4°, -4.5°,  0.0°] 位置[ 146.2,-379.1,1144.1] | 相机: ✓
记录点   3: 关节[ 37.1°,  4.9°,  1.3°,  0.5°, -4.4°,  0.0°] 位置[ 147.5,-380.2,1145.0] | 相机: ✓
...
```

### 4. 结束采集

按 `Ctrl+C` 停止，程序会自动：
- 停止拖拽模式
- 计算关节速度
- 保存数据为 pi0 格式
- 显示统计信息和下一步指引

## 📊 自动处理的数据转换

您的原始数据（角度制）：
```json
{
  "J1": 36.647,  // 度
  "J2": 4.704,
  ...
}
```

自动转换为 pi0 格式（弧度制 + 速度）：
```json
{
  "observation.state": [0.6397, 0.0821, ...],  // 弧度
  "action": [-0.0523, 0.1234, ..., 0.0]  // 关节速度(rad/s) + 夹爪
}
```

## 🎯 采集建议

### 一次采集应该包含什么？

- **完整任务轨迹**：从初始位置到完成任务
- **时长**：5-10 秒最佳（50-100 帧）
- **速度**：平滑自然，避免突然加速

### 需要多少数据？

| 任务复杂度 | 推荐采集次数 |
|-----------|------------|
| 简单（如抓取固定位置） | 10-20 次 |
| 中等（如抓取不同位置） | 30-50 次 |
| 复杂（如组装任务） | 50+ 次 |

### 如何采集多个 episode？

**方法 1：多次运行（推荐）**
```bash
# 第一次
python simple_drag_demo.py
# ... 拖拽示教 ...
# Ctrl+C 保存

# 第二次
python simple_drag_demo.py
# ... 拖拽示教 ...
# Ctrl+C 保存

# ... 重复 ...
```

每次运行会创建新的数据集文件夹。

**方法 2：批量合并**
```bash
# 采集多个数据集后，手动合并到一个目录
# 然后用转换脚本统一处理
```

## 📁 数据存储

### 目录结构

```
lerobot_dataset_20241014_153000/
├── data/
│   └── chunk-000/
│       └── file-000000.parquet   # 帧数据
├── meta/
│   ├── info.json                  # 数据集信息
│   ├── episodes.parquet           # Episode 元数据
│   ├── stats.json                 # 统计信息
│   └── tasks.parquet              # 任务列表
├── videos/
│   ├── observation.images.camera_color/
│   │   └── episode_000000.mp4
│   └── observation.images.camera_depth/
│       └── episode_000000.mp4
└── README.md
```

### 数据格式

**观测空间**：
- `observation.state`: 6 个关节位置（弧度）
- `observation.cartesian_position`: 末端笛卡尔坐标
- `observation.images.camera_color`: 彩色图像
- `observation.images.camera_depth`: 深度图像

**动作空间**：
- 7 维：6 个关节速度（rad/s）+ 1 个夹爪位置
- 自动从关节位置序列计算速度

## 🔧 配置修改

### 修改机器人 IP

编辑 `simple_drag_demo.py` 第 11 行：

```python
ROBOT_IP = "192.168.5.1"  # 改为您的机器人 IP
```

### 修改采集频率

编辑第 108 行：

```python
lerobot_saver = LeRobotDatasetSaver(
    repo_id=repo_id,
    fps=10,  # 改为您需要的频率（推荐 10 Hz）
    robot_type="dobot_cr5"
)
```

和第 257 行：

```python
time.sleep(0.1)  # 对应 10 Hz，如果改为 5 Hz 则改为 0.2
```

### 修改任务描述

编辑第 118 行：

```python
lerobot_saver.start_episode(task_name="teach_drag")  # 改为您的任务名称
```

例如：
```python
lerobot_saver.start_episode(task_name="pick_red_cube")
lerobot_saver.start_episode(task_name="place_in_box")
```

## ⚠️ 常见问题

### Q1: 程序启动后无法拖动机器人

**原因**：机器人未成功使能或拖拽模式未开启

**解决**：
```bash
# 检查程序输出是否有错误
# 确保看到：
# ✓ 使能机器人...
# ✓ 拖拽模式已开启!
```

### Q2: 相机初始化失败

```
⚠ 深度相机初始化失败，将只记录机器人数据
```

**解决**：
```bash
# 1. 检查相机连接
realsense-viewer

# 2. 检查 USB 权限
sudo chmod 666 /dev/bus/usb/*/*

# 3. 重新插拔相机

# 4. 如果不需要相机数据，可以忽略此警告
#    程序会继续记录机器人数据
```

### Q3: 数据保存位置在哪？

在 `CR5_TCP_test` 目录下：
```bash
ls -lh lerobot_dataset_*
```

### Q4: 如何验证数据格式？

```bash
# 检查数据集结构
tree lerobot_dataset_20241014_153000 -L 2

# 查看元数据
cat lerobot_dataset_20241014_153000/meta/info.json | jq

# 查看统计信息
cat lerobot_dataset_20241014_153000/meta/stats.json | jq
```

### Q5: 角度转弧度是否正确？

程序已自动处理！检查保存的数据：

```bash
# 打开 Python
python3

# 加载数据
import pandas as pd
df = pd.read_parquet('lerobot_dataset_xxx/data/chunk-000/file-000000.parquet')

# 查看关节位置（应该在 ±3.14 范围内）
print(df['observation.state'].iloc[0])
# 输出应该类似: [0.6397, 0.0821, 0.0187, 0.0064, -0.0794, 0.0]

# 查看关节速度
print(df['action'].iloc[0])
# 输出应该类似: [-0.0523, 0.1234, 0.0321, 0.0, -0.0123, 0.0, 0.0]
```

## 🚀 下一步：微调训练

### 采集多个 episodes 后

假设您已经运行了 10 次 `simple_drag_demo.py`，有 10 个数据集文件夹：

```bash
lerobot_dataset_20241014_153000/
lerobot_dataset_20241014_154000/
lerobot_dataset_20241014_155000/
...
```

### 方法 1：分别训练（推荐）

直接使用单个数据集微调：

```bash
cd /home/huang/learn_arm_robot/openpi

python scripts/train_pytorch.py \
    --pretrained-checkpoint pi05_droid \
    --dataset-repo-id ./examples/cr5/CR5_TCP_test/lerobot_dataset_20241014_153000 \
    --output-dir ./outputs/cr5_finetuned \
    --num-epochs 10 \
    --batch-size 8 \
    --use-lora \
    --lora-rank 8
```

### 方法 2：合并训练

如果想合并多个数据集：

```bash
# 1. 创建合并目录
mkdir -p combined_dataset/data
mkdir -p combined_dataset/videos/observation.images.camera_color
mkdir -p combined_dataset/videos/observation.images.camera_depth

# 2. 合并数据文件
# （这需要写脚本处理，或使用我们的转换工具）

# 3. 然后训练
python scripts/train_pytorch.py \
    --pretrained-checkpoint pi05_droid \
    --dataset-repo-id ./examples/cr5/combined_dataset \
    --output-dir ./outputs/cr5_finetuned \
    --num-epochs 20 \
    --batch-size 8 \
    --use-lora \
    --lora-rank 8
```

## 📖 技术细节

### 已修改的文件

1. **lerobot_dataset_saver.py**
   - ✅ 添加角度→弧度自动转换
   - ✅ 添加关节速度计算
   - ✅ 动作空间改为 7 维（6 速度 + 1 夹爪）

2. **simple_drag_demo.py**
   - ✅ 使用新的保存器初始化方式
   - ✅ 添加 pi0 格式说明输出

### 数据处理流程

```
原始数据采集
    ↓
[角度制, 10Hz]
    ↓
自动转换（add_frame）
    ↓
[弧度制存储]
    ↓
Episode 结束（save_episode）
    ↓
计算速度（_compute_joint_velocities）
    ↓
[速度动作]
    ↓
保存 Parquet（finalize_dataset）
    ↓
pi0 训练格式
```

## 🎉 总结

现在您只需：

```bash
# 1. 运行拖拽示教（重复 10-50 次）
python simple_drag_demo.py

# 2. 开始微调
cd /home/huang/learn_arm_robot/openpi
python scripts/train_pytorch.py \
    --pretrained-checkpoint pi05_droid \
    --dataset-repo-id ./examples/cr5/CR5_TCP_test/lerobot_dataset_xxx \
    --use-lora
```

就是这么简单！🚀
