#!/bin/bash

# 检查天翼云平台是否自动配置了分布式训练环境变量

echo "==================================="
echo "检查分布式训练环境变量"
echo "==================================="

echo "MASTER_ADDR: ${MASTER_ADDR:-未设置}"
echo "MASTER_PORT: ${MASTER_PORT:-未设置}"
echo "WORLD_SIZE: ${WORLD_SIZE:-未设置}"
echo "RANK: ${RANK:-未设置}"
echo "LOCAL_RANK: ${LOCAL_RANK:-未设置}"
echo "NNODES: ${NNODES:-未设置}"
echo "NODE_RANK: ${NODE_RANK:-未设置}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE:-未设置}"

echo ""
echo "==================================="
echo "NPU 设备信息"
echo "==================================="
python -c "import torch_npu; print(f'可用 NPU 数量: {torch_npu.npu.device_count()}')"

echo ""
echo "==================================="
echo "建议:"
echo "==================================="
if [ -z "$MASTER_ADDR" ]; then
    echo "⚠️  环境变量未自动配置"
    echo "   请在天翼云平台界面配置分布式训练参数"
    echo "   或使用 launch_qwen32b_multinode.sh 手动配置"
else
    echo "✅ 环境变量已自动配置"
    echo "   可以直接使用 launch_qwen32b_cloud.sh"
fi
