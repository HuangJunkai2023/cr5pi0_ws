#!/bin/bash
# CR5 真实机械臂控制 - 快速启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
ROBOT_IP="192.168.5.1"
SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
PROMPT="put the flash drive on the book"
ENABLE_GRIPPER=false
GRIPPER_PORT="/dev/ttyUSB0"
DRY_RUN=true  # 默认开启 DRY-RUN 模式（安全）
MAX_STEPS=100

# 交互式模式标志
INTERACTIVE_MODE=true

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

# 显示使用说明
show_help() {
    cat << EOF
CR5 真实机械臂控制 - 快速启动脚本

用法:
  $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -i, --robot-ip IP       CR5 机械臂 IP 地址 (默认: $ROBOT_IP)
  -s, --server HOST       策略服务器地址 (默认: $SERVER_HOST)
  -p, --port PORT         策略服务器端口 (默认: $SERVER_PORT)
  -t, --prompt TEXT       任务提示词 (默认: "$PROMPT")
  -g, --gripper           启用夹爪控制
  -d, --gripper-port PORT 夹爪串口 (默认: $GRIPPER_PORT)
  --real                  实际控制模式（关闭 DRY-RUN，实际控制机械臂）
  --max-steps N           最大控制步数 (默认: $MAX_STEPS)
  --no-interactive        禁用交互式选择，使用命令行参数

示例:
  # 交互式模式（默认，推荐）
  $0

  # DRY-RUN 测试（命令行模式）
  $0 --no-interactive --prompt "抓取红色方块"

  # 实际控制机械臂（命令行模式）
  $0 --no-interactive --real --prompt "抓取红色方块"

  # 启用夹爪 + 实际控制（命令行模式）
  $0 --no-interactive --real --gripper --gripper-port /dev/ttyUSB0 --prompt "抓取红色方块"

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--robot-ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        -s|--server)
            SERVER_HOST="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -t|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -g|--gripper)
            ENABLE_GRIPPER=true
            shift
            ;;
        -d|--gripper-port)
            GRIPPER_PORT="$2"
            shift 2
            ;;
        --real)
            DRY_RUN=false
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --no-interactive)
            INTERACTIVE_MODE=false
            shift
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 交互式选择模式
if [ "$INTERACTIVE_MODE" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         CR5 真实机械臂控制程序 - 交互式启动                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "请选择运行模式："
    echo ""
    echo "  ${GREEN}[y]${NC} 实机测试模式"
    echo "      - 实际控制 CR5 机械臂"
    echo "      - 启用夹爪控制"
    echo "      - 任务: put the flash drive on the book"
    echo ""
    echo "  ${BLUE}[n]${NC} 虚拟测试模式（DRY-RUN）"
    echo "      - 只打印命令，不实际控制机械臂"
    echo "      - 带摄像头采集真实图像"
    echo "      - 任务: put the flash drive on the book"
    echo ""
    read -p "请输入选择 [y/n]: " choice
    echo ""
    
    case "$choice" in
        y|Y|yes|YES)
            print_success "选择：实机测试模式"
            DRY_RUN=false
            ENABLE_GRIPPER=true
            PROMPT="put the flash drive on the book"
            ;;
        n|N|no|NO)
            print_success "选择：虚拟测试模式（DRY-RUN）"
            DRY_RUN=true
            ENABLE_GRIPPER=false
            PROMPT="put the flash drive on the book"
            ;;
        *)
            print_error "无效的选择: $choice"
            print_warning "默认使用虚拟测试模式（DRY-RUN）"
            DRY_RUN=true
            ENABLE_GRIPPER=false
            PROMPT="put the flash drive on the book"
            ;;
    esac
    echo ""
fi

# 打印配置
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         CR5 真实机械臂控制程序 - 启动配置                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
print_info "机械臂 IP:      $ROBOT_IP"
print_info "策略服务器:      $SERVER_HOST:$SERVER_PORT"
print_info "任务提示:        $PROMPT"
print_info "DRY-RUN 模式:   $DRY_RUN"
print_info "最大步数:        $MAX_STEPS"
print_info "启用夹爪:        $ENABLE_GRIPPER"
if [ "$ENABLE_GRIPPER" = true ]; then
    print_info "夹爪串口:        $GRIPPER_PORT"
fi
echo ""

# 如果是 DRY-RUN 模式，显示警告
if [ "$DRY_RUN" = true ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  ⚠️  DRY-RUN 模式                              ║"
    echo "║           程序将只打印数据，不会实际控制机械臂                  ║"
    echo "║           但会使用真实摄像头采集图像进行推理                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  ⚠️  实机控制模式                              ║"
    echo "║               程序将实际控制 CR5 机械臂！                      ║"
    echo "║           请确保已做好充分的安全准备！                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
fi

# 检查策略服务器是否运行
print_info "检查策略服务器连接..."
if timeout 3 bash -c "echo > /dev/tcp/$SERVER_HOST/$SERVER_PORT" 2>/dev/null; then
    print_success "策略服务器已连接 ✓"
else
    print_error "无法连接到策略服务器 $SERVER_HOST:$SERVER_PORT"
    echo ""
    print_warning "请先启动策略服务器："
    echo "  cd /home/huang/learn_arm_robot/openpi"
    echo "  uv run scripts/serve_policy.py \\"
    echo "      --checkpoint checkpoints/pi0_cr5_finetune_lora \\"
    echo "      --env DROID"
    exit 1
fi

# 检查相机
print_info "检查 RealSense 相机..."
if command -v rs-enumerate-devices &> /dev/null; then
    if rs-enumerate-devices 2>/dev/null | grep -q "Device info"; then
        print_success "RealSense 相机已找到 ✓"
    else
        print_warning "未检测到 RealSense 相机"
        print_warning "程序将继续运行，但可能无法获取图像"
    fi
else
    print_warning "未安装 realsense-tools，跳过相机检查"
fi

# 检查夹爪串口（如果启用）
if [ "$ENABLE_GRIPPER" = true ]; then
    print_info "检查夹爪串口..."
    if [ -e "$GRIPPER_PORT" ]; then
        print_success "夹爪串口已找到 ✓"
    else
        print_error "夹爪串口不存在: $GRIPPER_PORT"
        print_warning "可用的串口设备："
        ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  (无)"
        exit 1
    fi
fi

echo ""
print_info "准备启动控制程序..."
echo ""

# 根据模式显示不同的安全提示
if [ "$DRY_RUN" = true ]; then
    print_info "📷  虚拟测试模式准备："
    echo "  1. 确保 RealSense 相机已连接"
    echo "  2. 将场景设置为: 闪存盘和书本"
    echo "  3. 相机对准工作区域"
    echo "  4. 观察打印的控制命令是否合理"
    echo ""
else
    print_warning "⚠️  实机控制安全提示："
    echo "  1. ${RED}确保急停按钮触手可及${NC}"
    echo "  2. 清空机械臂工作空间"
    echo "  3. 人员与机械臂保持安全距离（至少 2 米）"
    echo "  4. 机械臂应处于 TCP 控制模式"
    echo "  5. 夹爪已正确连接并校准"
    echo "  6. 场景设置: 闪存盘放在合适位置，书本作为目标"
    echo ""
fi

# 等待用户确认
read -p "按 Enter 继续启动，按 Ctrl+C 取消..."
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_SCRIPT="$SCRIPT_DIR/cr5_real_control.py"

# 检查控制脚本是否存在
if [ ! -f "$CONTROL_SCRIPT" ]; then
    print_error "控制脚本不存在: $CONTROL_SCRIPT"
    exit 1
fi

# 构建命令
CMD="uv run $CONTROL_SCRIPT"
CMD="$CMD --robot-ip $ROBOT_IP"
CMD="$CMD --server-host $SERVER_HOST"
CMD="$CMD --server-port $SERVER_PORT"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --max-steps $MAX_STEPS"

# DRY-RUN 模式控制
if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
else
    CMD="$CMD --no-dry-run"
fi

if [ "$ENABLE_GRIPPER" = true ]; then
    CMD="$CMD --enable-gripper"
    CMD="$CMD --gripper-port $GRIPPER_PORT"
fi

# 打印完整命令
print_info "执行命令:"
echo "  $CMD"
echo ""

# 执行命令
eval $CMD

# 退出后的清理
echo ""
if [ $? -eq 0 ]; then
    print_success "程序正常退出"
else
    print_error "程序异常退出"
fi
