#!/bin/bash
# CTyunOS 22.06.2 训练脚本 - Qwen2.5-32B-Instruct + 8张华为昇腾910B
# 系统：CTyunOS 22.06.2@ascend-910b 64位
# 硬件：8*HuaweiAscend 910B
# 使用 DeepSpeed ZeRO-3 优化（32B模型单卡显存不足，必须使用ZeRO-3分片参数）

# ============================================
# 天翼云训推服务 - Qwen-32B 标准微调（无课程学习）
# 适用于平台自动配置环境变量的场景
# ============================================

# ---------- Python 环境配置 ----------
# 优先使用系统 Python（已安装 torch-npu），如需切换请修改此处
# 可通过环境变量覆盖：PYTHON_BIN=/path/to/python bash launch_qwen32b_standard.sh
PYTHON_BIN="${PYTHON_BIN:-/root/PERL-FORK/venv_perl/bin/python}"

# 验证 torch 可用性
if ! "${PYTHON_BIN}" -c "import torch" 2>/dev/null; then
    echo "[ERROR] ${PYTHON_BIN} 无法导入 torch，请检查 Python 环境"
    echo "[HINT]  当前 python 路径: $(which python)"
    echo "[HINT]  尝试: PYTHON_BIN=/path/to/correct/python bash launch_qwen32b_standard.sh"
    exit 1
fi
echo "[INFO] Python 路径: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[INFO] torch 版本: $(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')"

# ---------- 路径配置（按需修改）----------
MODEL_PATH="${MODEL_PATH:-/mnt/nvme0/models/Qwen2.5-32B-Instruct}"
BASE_MODEL="$MODEL_PATH"
DATA_PATH="/root/PERL-FORK/ft-dataset/kicad_sft_dataset_590.json"
OUTPUT_DIR=/mnt/nvme0/output/grpo_lora_qwen25_32b_ctyunos_910b_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p "${OUTPUT_DIR}"

# 激活 GCC 10（CPUAdam 编译需要 GCC 9+）
export PATH="/opt/ctyunos/gcc-toolset-10/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/opt/ctyunos/gcc-toolset-10/root/usr/lib64:$LD_LIBRARY_PATH"
echo "[INFO] GCC 版本: $(gcc --version | head -1)"

# 设置NPU环境变量（CTyunOS专用）
# source CANN 环境，确保 TBE/ACL 组件正确加载
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "[INFO] CANN 环境已加载"
else
    echo "[WARN] 未找到 set_env.sh，跳过 CANN 环境初始化"
fi

export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export COMBINED_ENABLE=1       # 启用混合精度优化
export TASK_QUEUE_ENABLE=1     # 启用任务队列优化

# Qwen2.5-32B 显存优化：启用梯度检查点相关优化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

echo "[INFO] 使用模型路径: ${MODEL_PATH}"
echo "[INFO] 数据集路径: ${DATA_PATH}"
echo "[INFO] 输出目录: ${OUTPUT_DIR}"
echo "[INFO] 日志文件: ${LOG_FILE}"
echo "📊 Environment Info:"
echo "   - MASTER_ADDR: ${MASTER_ADDR:-auto}"
echo "   - MASTER_PORT: ${MASTER_PORT:-auto}"
echo "   - WORLD_SIZE: ${WORLD_SIZE:-auto}"
echo "   - RANK: ${RANK:-auto}"
echo "   - LOCAL_RANK: ${LOCAL_RANK:-auto}"
echo "[INFO] 开始训练..."

# 训练超参数
BATCH_SIZE=128
MICRO_BATCH_SIZE=1
NUM_EPOCHS=3
LEARNING_RATE=2e-5
CUTOFF_LEN=512

# LoRA 参数
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

# DeepSpeed 配置文件（ZeRO-3，32B模型必须分片参数到所有卡）
DS_CONFIG="./ds_config_zero3.json"

# 直接运行训练脚本
/root/PERL-FORK/venv_perl/bin/torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    finetune_npu_deepspeed_standard.py \
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
    --deepspeed_config "$DS_CONFIG" \
    2>&1 | tee "${LOG_FILE}"

# ============================================
# 使用说明:
#
# 1. 在天翼云训推服务网页界面:
#    - 选择训练任务类型: 分布式训练
#    - 节点数: 1
#    - 每节点 NPU 数: 8
#    - 启动脚本: bash launch_qwen32b_standard.sh
#
# 2. 平台会自动在每个节点上执行此脚本
# 3. 环境变量由平台自动注入
#
# 注意: 此版本不包含课程学习功能，适用于标准数据集
# ============================================
