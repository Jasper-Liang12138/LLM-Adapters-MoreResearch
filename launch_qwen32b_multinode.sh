#!/bin/bash

# ============================================
# Qwen-32B 多节点训练启动脚本
# 硬件: 2 节点 × 8 × Ascend 910B = 16 NPUs
# 框架: DeepSpeed ZeRO-3
# ============================================

# 节点配置
export MASTER_ADDR="<主节点IP>"  # 替换为实际的主节点 IP
export MASTER_PORT=29500
export NNODES=2  # 节点数
export NODE_RANK=${NODE_RANK:-0}  # 当前节点编号 (0 或 1)
export NPROC_PER_NODE=8  # 每个节点的 NPU 数量

# 模型和数据路径
BASE_MODEL="Qwen/Qwen-32B"  # 或本地路径
DATA_PATH="/work/basicData/2021154936252485632"
OUTPUT_DIR="/work/mount/Qwen32bLoraSft"

# 训练超参数
BATCH_SIZE=128
MICRO_BATCH_SIZE=1  # ZeRO-3 建议使用 1
NUM_EPOCHS=3
LEARNING_RATE=2e-5
CUTOFF_LEN=2048

# LoRA 参数
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# Curriculum Learning 参数
USE_CURRICULUM=True  # 启用课程学习
CURRICULUM_SEED=42

# DeepSpeed 配置文件
DS_CONFIG="./ds_config_zero3.json"

# 启动训练
deepspeed --num_nodes=$NNODES \
          --num_gpus=$NPROC_PER_NODE \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          --node_rank=$NODE_RANK \
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
          --use_curriculum $USE_CURRICULUM \
          --curriculum_seed $CURRICULUM_SEED \
          --deepspeed_config "$DS_CONFIG"

# ============================================
# 使用说明:
#
# 1. 在主节点 (NODE_RANK=0) 上运行:
#    export NODE_RANK=0
#    bash launch_qwen32b_multinode.sh
#
# 2. 在从节点 (NODE_RANK=1) 上运行:
#    export NODE_RANK=1
#    bash launch_qwen32b_multinode.sh
#
# 3. 确保两个节点可以通过 SSH 免密登录
# 4. 确保 MASTER_ADDR 设置为主节点的 IP 地址
# ============================================
