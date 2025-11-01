# CR5 机械臂推理客户端使用指南

这个客户端用于连接 OpenPI 推理服务器，获取策略动作并控制 CR5 机械臂执行。

## 功能特性

- ✅ 连接到 OpenPI 推理服务器
- ✅ 实时获取机械臂状态（关节位置、速度）
- ✅ 实时获取 RealSense 相机图像
- ✅ 控制 Robotiq 2F-85 夹爪
- ✅ 执行推理服务器返回的动作
- ✅ 可配置的控制频率

## 系统要求

### 硬件
- CR5 机械臂（通过网络连接）
- RealSense 相机（D435/D455 等）
- Robotiq 2F-85 夹爪（可选）

### 软件依赖
- Python 3.8+
- openpi-client
- pyrealsense2
- numpy
- opencv-python
- tyro

## 安装

1. 确保已经安装了 OpenPI 项目的依赖：
```bash
cd /home/huang/learn_arm_robot/openpi
uv sync
```

2. 安装额外的依赖：
```bash
pip install pyrealsense2
```

3. 确保 CR5 API 文件在正确的位置：
   - `examples/cr5/CR5_TCP_test/files/dobot_api.py`
   - `examples/cr5/CR5_TCP_test/files/gripper.py`（如果使用夹爪）

## 使用方法

### 第 1 步：启动推理服务器

在一个终端窗口中启动推理服务器：

```bash
# 示例：启动 CR5 推理服务器
uv run scripts/serve_policy.py \
    --checkpoint_path checkpoints/pi0_cr5_finetune_lora/cr5_test_dataset \
    --host 0.0.0.0 \
    --port 8000
```

确认服务器启动成功，看到类似以下输出：
```
INFO:root:Creating server (host: huang-laptop, ip: 127.0.1.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### 第 2 步：运行客户端

在另一个终端窗口中运行客户端：

```bash
cd /home/huang/learn_arm_robot/openpi

# 基本用法（使用默认参数）
uv run examples/cr5/cr5_client.py

# 指定服务器地址
uv run examples/cr5/cr5_client.py \
    --host 127.0.1.1 \
    --port 8000

# 完整参数示例
uv run examples/cr5/cr5_client.py \
    --host 127.0.1.1 \
    --port 8000 \
    --robot_ip 192.168.5.1 \
    --camera_width 640 \
    --camera_height 480 \
    --gripper_port /dev/ttyUSB0 \
    --num_episodes 5 \
    --control_frequency 10.0 \
    --prompt "put the flash drive on the book"
```

### 第 3 步：执行任务

客户端启动后会：
1. 连接到推理服务器
2. 初始化 CR5 机械臂
3. 启动相机
4. 初始化夹爪（如果配置）
5. 预热推理服务器
6. 等待你按 Enter 开始执行每个 episode

按 `Ctrl+C` 可以随时停止执行。

## 命令行参数

### 推理服务器配置
- `--host`: 推理服务器地址（默认: `127.0.1.1`）
- `--port`: 推理服务器端口（默认: `8000`）
- `--api_key`: API 密钥（可选）

### CR5 机械臂配置
- `--robot_ip`: CR5 机械臂 IP 地址（默认: `192.168.5.1`）

### 相机配置
- `--camera_width`: 相机宽度（默认: `640`）
- `--camera_height`: 相机高度（默认: `480`）
- `--camera_fps`: 相机帧率（默认: `30`）

### 夹爪配置
- `--gripper_port`: 夹爪串口（默认: `/dev/ttyUSB0`，设为 `None` 跳过夹爪）

### 执行配置
- `--num_episodes`: 执行的 episode 数量（默认: `10`）
- `--control_frequency`: 控制频率 Hz（默认: `10.0`）

### 任务描述
- `--prompt`: 任务描述（默认: `"put the flash drive on the book"`）

## 重要配置说明

### 观测数据格式

客户端会将以下数据发送给推理服务器：

```python
observation = {
    "state": np.array([...]),  # 状态向量（7 维：6 关节位置 + 1 夹爪）
    "images": {
        "cam_high": np.array([...]),  # 图像 (3, H, W)
    },
    "prompt": "任务描述",
}
```

**⚠️ 重要**: 确保这个格式与你训练模型时使用的格式一致！

如果你的训练配置不同，需要修改 `create_observation()` 函数：
- 调整状态向量的维度和内容
- 调整图像的键名（如 `cam_high`, `cam_low`, `cam_wrist` 等）
- 添加或删除其他观测数据

### 动作数据格式

客户端期望推理服务器返回的动作格式：

```python
action = {
    "action": np.array([...]),  # 动作向量（7 维：6 关节位置 + 1 夹爪）
    # 其他可选字段...
}
```

**⚠️ 重要**: 如果你的模型输出格式不同，需要修改 `execute_action()` 函数。

## 故障排查

### 1. 无法连接推理服务器

**问题**: `ConnectionRefusedError` 或 `WebSocket connection failed`

**解决方案**:
- 确认推理服务器已启动
- 检查 `--host` 和 `--port` 参数是否正确
- 如果服务器在远程主机，确保防火墙允许端口访问

### 2. 无法连接 CR5 机械臂

**问题**: `⚠️  无法获取机械臂状态`

**解决方案**:
- 确认机械臂已开机并连接到网络
- 检查 `--robot_ip` 参数是否正确
- 使用 `ping 192.168.5.1` 测试网络连接
- 确认没有其他程序占用机械臂连接

### 3. 无法启动相机

**问题**: `⚠️  获取图像失败`

**解决方案**:
- 确认 RealSense 相机已连接
- 使用 `realsense-viewer` 测试相机
- 检查 USB 连接和驱动

### 4. 夹爪初始化失败

**问题**: `⚠️  夹爪初始化失败`

**解决方案**:
- 确认夹爪已连接到正确的串口
- 检查串口权限: `sudo chmod 666 /dev/ttyUSB0`
- 使用 `ls /dev/ttyUSB*` 或 `ls /dev/ttyACM*` 查找串口
- 如果不使用夹爪，设置 `--gripper_port None`

### 5. 观测/动作格式不匹配

**问题**: 推理正常但动作执行异常

**解决方案**:
- 检查 `create_observation()` 函数中的观测格式
- 检查 `execute_action()` 函数中的动作解析
- 查看训练时的数据格式配置
- 添加日志输出调试数据形状和类型

## 自定义修改

### 修改观测数据格式

编辑 `cr5_client.py` 中的 `create_observation()` 函数：

```python
def create_observation(robot, camera, prompt):
    # 1. 修改状态向量
    state = np.concatenate([
        joint_pos,        # 6 维
        joint_vel,        # 6 维（如果需要）
        [gripper_pos],    # 1 维
    ])
    
    # 2. 修改图像格式
    observation = {
        "state": state,
        "images": {
            "cam_high": image_chw,      # 主相机
            "cam_wrist": wrist_image,   # 腕部相机（如果有）
            # 添加更多相机...
        },
        "prompt": prompt,
    }
    
    return observation
```

### 修改动作执行逻辑

编辑 `cr5_client.py` 中的 `execute_action()` 函数：

```python
def execute_action(robot, gripper, action):
    action_data = action.get("action", None)
    
    # 根据你的动作格式解析
    if len(action_data) == 7:
        # 绝对位置控制
        joint_positions = action_data[:6]
        gripper_position = action_data[6]
    elif len(action_data) == 12:
        # 位置 + 速度控制
        joint_positions = action_data[:6]
        joint_velocities = action_data[6:12]
        # ... 实现速度控制
    
    # 执行动作
    robot.move_joints(joint_positions)
    gripper.set_position(gripper_position)
```

### 添加完成条件

编辑主循环中的完成检测：

```python
# 在主循环中
if step_count >= 100:  # 简单的步数限制
    break

# 或者添加更复杂的条件
if is_task_completed(observation, action):
    break
```

## 性能优化

1. **调整控制频率**: 使用 `--control_frequency` 参数
   - 10 Hz: 适合大多数任务
   - 20 Hz: 需要更快响应的任务
   - 5 Hz: 计算资源受限时

2. **降低图像分辨率**: 使用 `--camera_width` 和 `--camera_height`
   - 训练时使用多大分辨率，推理时就使用多大

3. **GPU 加速**: 确保推理服务器使用 GPU

## 安全提示

⚠️ **使用机械臂时请注意安全**：

1. 首次运行时，确保机械臂周围没有障碍物和人员
2. 准备好急停按钮
3. 从低速度开始测试（调整 `robot.move_joints()` 中的 `speed` 参数）
4. 监控机械臂运动，发现异常立即按 `Ctrl+C` 停止
5. 确保机械臂工作空间安全

## 相关文件

- `cr5_client.py`: 主客户端程序
- `CR5_TCP_test/dobot_api.py`: CR5 API
- `CR5_TCP_test/gripper.py`: 夹爪控制
- `CR5_TCP_test/lerobot_collect.py`: 数据采集参考

## 参考

- [OpenPI 项目](https://github.com/Physical-Intelligence/openpi)
- [LeRobot 文档](https://github.com/huggingface/lerobot)
- [CR5 用户手册](https://www.dobot.cc/)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)
