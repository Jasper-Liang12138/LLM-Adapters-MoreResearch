#!/bin/bash

# ============================================
# Qwen-32B 单节点训练脚本 (用于测试)
# 硬件: 1 节点 × 8 × Ascend 910B
# ============================================

export NPROC_PER_NODE=8

BASE_MODEL="Qwen/Qwen-32B"
DATA_PATH="/path/to/your/dataset.json"
OUTPUT_DIR="./trained_models/qwen32b-lora-test"

deepspeed --num_gpus=$NPROC_PER_NODE \
          finetune_npu_deepspeed.py \
          --base_model "$BASE_MODEL" \
          --data_path "$DATA_PATH" \
          --output_dir "$OUTPUT_DIR" \
          --adapter_name lora \
          --batch_size 64 \
          --micro_batch_size 1 \
          --num_epochs 1 \
          --learning_rate 2e-5 \
          --cutoff_len 2048 \
          --lora_r 64 \
          --lora_alpha 128 \
          --target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
          --use_curriculum True \
          --curriculum_seed 42 \
          --deepspeed_config "./ds_config_zero3.json"
