#!/usr/bin/env python3
"""
CR5 客户端简单测试脚本
用于测试推理服务器连接和基本功能
"""

import sys
import time
from pathlib import Path

import numpy as np

# 添加 openpi-client 路径
client_path = Path(__file__).parent.parent.parent / "packages" / "openpi-client" / "src"
sys.path.insert(0, str(client_path))

from openpi_client import websocket_client_policy

def test_server_connection(host="127.0.1.1", port=8000):
    """测试服务器连接"""
    print("=" * 60)
    print("测试推理服务器连接")
    print("=" * 60)
    
    try:
        print(f"🌐 连接到 {host}:{port}...")
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=None,
        )
        print("✅ 连接成功！")
        
        # 获取服务器元数据
        metadata = policy.get_server_metadata()
        print(f"📊 服务器元数据:")
        print(f"   {metadata}")
        
        return policy
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None


def test_inference(policy):
    """测试推理功能"""
    print("\n" + "=" * 60)
    print("测试推理功能")
    print("=" * 60)
    
    try:
        # 创建模拟观测数据
        # CR5 格式：state (6维，只有关节位置), image (单张图像), prompt (可选)
        observation = {
            "state": np.random.rand(6).astype(np.float32),  # 6 维状态（关节位置）
            "image": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "prompt": "test inference",
        }
        
        print("🔥 发送观测数据...")
        print(f"   状态形状: {observation['state'].shape}")
        print(f"   图像形状: {observation['image'].shape}")
        
        # 执行推理
        start_time = time.time()
        action = policy.infer(observation)
        elapsed = time.time() - start_time
        
        print(f"✅ 推理成功！耗时: {elapsed*1000:.1f} ms")
        print(f"📦 动作数据:")
        
        # 显示动作信息
        if isinstance(action, dict):
            for key, value in action.items():
                if isinstance(value, np.ndarray):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"   {key}: {value}")
        else:
            print(f"   类型: {type(action)}")
            if isinstance(action, np.ndarray):
                print(f"   形状: {action.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_inferences(policy, num_iterations=10):
    """测试多次推理并统计性能"""
    print("\n" + "=" * 60)
    print(f"测试多次推理 (n={num_iterations})")
    print("=" * 60)
    
    try:
        observation = {
            "state": np.random.rand(6).astype(np.float32),  # 6 维状态
            "image": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "prompt": "test multiple inferences",
        }
        
        timings = []
        
        print("🔄 执行推理...")
        for i in range(num_iterations):
            start_time = time.time()
            action = policy.infer(observation)
            elapsed = time.time() - start_time
            timings.append(elapsed * 1000)  # 转换为毫秒
            
            if (i + 1) % 5 == 0:
                print(f"   完成 {i + 1}/{num_iterations}")
        
        # 统计
        timings = np.array(timings)
        print(f"\n📊 性能统计:")
        print(f"   平均: {timings.mean():.1f} ms")
        print(f"   标准差: {timings.std():.1f} ms")
        print(f"   最小: {timings.min():.1f} ms")
        print(f"   最大: {timings.max():.1f} ms")
        print(f"   中位数: {np.median(timings):.1f} ms")
        print(f"   频率: {1000 / timings.mean():.1f} Hz")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("\n🤖 CR5 客户端测试工具\n")
    
    # 解析命令行参数
    host = "127.0.1.1"
    port = 8000
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print(f"配置:")
    print(f"  服务器: {host}:{port}")
    print()
    
    # 测试 1: 连接服务器
    policy = test_server_connection(host, port)
    if policy is None:
        print("\n❌ 无法连接到服务器，请检查:")
        print("  1. 服务器是否已启动")
        print("  2. 主机地址和端口是否正确")
        print("  3. 防火墙设置")
        return
    
    # 测试 2: 单次推理
    success = test_inference(policy)
    if not success:
        print("\n❌ 推理测试失败，请检查:")
        print("  1. 观测数据格式是否与模型匹配")
        print("  2. 模型是否正确加载")
        return
    
    # 测试 3: 多次推理
    test_multiple_inferences(policy, num_iterations=20)
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  运行完整客户端: python examples/cr5/cr5_client.py --help")


if __name__ == "__main__":
    main()
