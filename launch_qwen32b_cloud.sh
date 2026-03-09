#!/bin/bash

# ============================================
# 天翼云训推服务 - Qwen-32B 多节点训练
# 适用于平台自动配置环境变量的场景
# ============================================

# 模型和数据路径
BASE_MODEL="/work/mount/publicModel/Qwen3-32B"  # 或本地路径
DATA_PATH="LLM-Adapters-MoreResearch/dataset/math_10k.json"
OUTPUT_DIR="/work/mount/Output/Qwen32bLoraSft"

# 训练超参数
BATCH_SIZE=128
MICRO_BATCH_SIZE=1
NUM_EPOCHS=3
LEARNING_RATE=2e-5
CUTOFF_LEN=2048

# LoRA 参数
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# DeepSpeed 配置文件
DS_CONFIG="./ds_config_zero3.json"

# ============================================
# 平台会自动设置以下环境变量，无需手动配置:
# - MASTER_ADDR
# - MASTER_PORT
# - WORLD_SIZE
# - RANK
# - LOCAL_RANK
# ============================================

echo "🚀 Starting training on 天翼云..."
echo "📊 Environment Info:"
echo "   - MASTER_ADDR: ${MASTER_ADDR:-auto}"
echo "   - MASTER_PORT: ${MASTER_PORT:-auto}"
echo "   - WORLD_SIZE: ${WORLD_SIZE:-auto}"
echo "   - RANK: ${RANK:-auto}"
echo "   - LOCAL_RANK: ${LOCAL_RANK:-auto}"

# 平台 WORLD_SIZE = 总进程数（节点数 × 每节点卡数），需换算为节点数
NPROC_PER_NODE=8
NNODES=$(( ${WORLD_SIZE:-8} / NPROC_PER_NODE ))

# 直接运行训练脚本（DeepSpeed 会读取环境变量）
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-23456} \
    --node_rank=${RANK:-0} \
    finetune_npu_deepspeed.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --adapter_name lora \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --cutoff_len $CUTOFF_LEN \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --train_on_inputs False \
    --deepspeed_config "$DS_CONFIG"

# ============================================
# 使用说明:
#
# 1. 在天翼云训推服务网页界面:
#    - 选择训练任务类型: 分布式训练
#    - 节点数: 2
#    - 每节点 NPU 数: 8
#    - 启动脚本: bash launch_qwen32b_cloud.sh
#
# 2. 平台会自动在每个节点上执行此脚本
# 3. 环境变量由平台自动注入
# ============================================
