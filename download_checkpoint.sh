#!/bin/bash

# 检查点文件下载脚本 - 完整版本（使用代理）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

BASE_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim"
BASE_URL="https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o"

cd "$BASE_DIR"

echo "开始下载完整的 pi0_aloha_sim 检查点文件..."
echo "使用代理: $https_proxy"
echo "这可能需要一些时间，因为文件总大小约为 9GB+"

# 下载小文件
echo "1. 下载小参数文件..."
curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/30c0844499e9a74b299b4acf1d892b16" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F30c0844499e9a74b299b4acf1d892b16?generation=1748456809409134&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/4bf4bb1583b53cf4763eb2c076cfe909" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F4bf4bb1583b53cf4763eb2c076cfe909?generation=1748456806802252&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/d526dfedf24bfc5e856a5c9dce64f4dc" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2Fd526dfedf24bfc5e856a5c9dce64f4dc?generation=1748456809100156&alt=media"

# 下载中等大小的文件
echo "2. 下载中等参数文件 (~200MB)..."
curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/3492325ff87f75fe731e3f0e1568a284" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F3492325ff87f75fe731e3f0e1568a284?generation=1748456810189211&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/ff7729acbdf8e670028ac31100202c9f" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2Fff7729acbdf8e670028ac31100202c9f?generation=1748456810806928&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/04e1abe7d7bee0b72bc2ccc4acd51e6d" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F04e1abe7d7bee0b72bc2ccc4acd51e6d?generation=1748456825885846&alt=media"

# 下载大文件 (~1GB 级别)
echo "3. 下载大型参数文件 (~1GB 每个，可能需要几分钟)..."
curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/38f5f89a89ba4de6f36dae38afb812df" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F38f5f89a89ba4de6f36dae38afb812df?generation=1748456838887130&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/5a69fee77c2cf84d3a2b35d9ea1d024b" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F5a69fee77c2cf84d3a2b35d9ea1d024b?generation=1748456834123363&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/89c4de373fec21a170a2c1e8260887d1" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F89c4de373fec21a170a2c1e8260887d1?generation=1748456837763694&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/a0854df81ac9825231de7438b7f190e0" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2Fa0854df81ac9825231de7438b7f190e0?generation=1748456848130774&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/b7ffa1e5fe2c4d883c7630d8eab57645" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2Fb7ffa1e5fe2c4d883c7630d8eab57645?generation=1748456867077742&alt=media"

# 下载超大文件 (~2GB 级别)
echo "4. 下载超大参数文件 (~2GB 每个，需要较长时间)..."
curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/a2581300d76d041a128b3655b355e2e0" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2Fa2581300d76d041a128b3655b355e2e0?generation=1748456867235149&alt=media"

curl -L --proxy $https_proxy -o "params/ocdbt.process_0/d/3454ef21b44daf9a80117d2205930dfc" "$BASE_URL/checkpoints%2Fpi0_aloha_sim%2Fparams%2Focdbt.process_0%2Fd%2F3454ef21b44daf9a80117d2205930dfc?generation=1748456868624983&alt=media"

echo "所有检查点文件下载完成！"
echo "总下载大小: 约 9GB+"
echo "现在可以启动服务器了: uv run scripts/serve_policy.py --env ALOHA_SIM"
