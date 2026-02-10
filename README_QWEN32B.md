# Qwen-32B NPU 多节点训练指南

## 环境配置

- **硬件**: 2 节点 × 8 × Ascend 910B = 16 NPUs
- **镜像**: ubuntu22.04-teleformers-cann8.2.rc1-npu v0.2.0.post1.ssh
- **框架**: HuggingFace Transformers + DeepSpeed ZeRO-3
- **模型**: Qwen-32B

## 文件说明

```
├── finetune_npu_deepspeed.py      # 主训练脚本
├── ds_config_zero3.json           # DeepSpeed ZeRO-3 配置
├── launch_qwen32b_multinode.sh    # 多节点启动脚本
├── launch_qwen32b_single.sh       # 单节点测试脚本
└── README_QWEN32B.md              # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install \
    "transformers==4.51.3" \
    datasets \
    accelerate \
    deepspeed \
    bitsandbytes \
    sentencepiece \
    tokenizers

# 确保 torch_npu 已安装
python -c "import torch_npu; print(torch_npu.__version__)"
```

**注意**: transformers 4.51.3 应该是镜像自带的版本，如果已安装则无需重复安装。

### 2. 准备数据

#### 标准数据格式 (JSON):
```json
[
  {
    "instruction": "问题描述",
    "input": "可选的额外输入",
    "output": "期望的输出"
  },
  ...
]
```

#### 课程学习数据格式 (带 difficulty 字段):
```json
[
  {
    "instruction": "简单的问题",
    "input": "",
    "output": "答案",
    "difficulty": "explain"
  },
  {
    "instruction": "需要推理的问题",
    "input": "",
    "output": "推理过程和答案",
    "difficulty": "reasoning"
  },
  {
    "instruction": "复杂的拓扑问题",
    "input": "",
    "output": "详细分析",
    "difficulty": "topology"
  }
]
```

**Difficulty 级别说明**:
- `explain`: 解释性内容，难度最低
- `reasoning`: 需要推理的内容，中等难度
- `topology`: 复杂的拓扑/结构化内容，难度最高

### 3. 单节点测试 (推荐先测试)

```bash
# 修改 launch_qwen32b_single.sh 中的路径
vim launch_qwen32b_single.sh

# 运行
bash launch_qwen32b_single.sh
```

### 4. 多节点训练

#### 步骤 1: 配置主节点 IP

编辑 `launch_qwen32b_multinode.sh`:
```bash
export MASTER_ADDR="192.168.1.100"  # 替换为实际主节点 IP
```

#### 步骤 2: 在主节点启动

```bash
export NODE_RANK=0
bash launch_qwen32b_multinode.sh
```

#### 步骤 3: 在从节点启动

```bash
export NODE_RANK=1
bash launch_qwen32b_multinode.sh
```

## DeepSpeed ZeRO-3 配置说明

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage` | 3 | ZeRO 阶段 3，分片参数、梯度和优化器状态 |
| `offload_optimizer` | CPU | 优化器状态卸载到 CPU |
| `offload_param` | CPU | 参数卸载到 CPU |
| `train_micro_batch_size_per_gpu` | 1 | 每个 NPU 的 micro batch size |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |

### 内存优化

对于 Qwen-32B (约 64GB):
- **ZeRO-3 + CPU Offload**: 可在 16×910B (每卡 64GB HBM) 上训练
- **有效 Batch Size**: `1 × 8 × 16 = 128`

## 超参数建议

### 课程学习 (Curriculum Learning)

**启用课程学习** (推荐):
```bash
--use_curriculum True
--curriculum_seed 42
```

**课程学习策略**:
- **Early Stage (0-30%)**: 80% explain + 20% reasoning + 0% topology
- **Mid Stage (30-70%)**: 40% explain + 40% reasoning + 20% topology
- **Late Stage (70-100%)**: 25% explain + 35% reasoning + 40% topology

**优势**:
- 从简单到复杂的渐进式学习
- 提高模型收敛速度
- 减少训练不稳定性
- 更好的泛化能力

**注意**: 如果数据集没有 `difficulty` 字段，课程学习会自动禁用，回退到标准训练。

### LoRA 参数

```python
lora_r = 64              # Rank (32B 模型建议 64-128)
lora_alpha = 128         # Alpha (通常是 r 的 2 倍)
lora_dropout = 0.05      # Dropout
target_modules = [       # 目标模块
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]
```

### 训练参数

```python
learning_rate = 2e-5     # 大模型建议 1e-5 到 5e-5
num_epochs = 3           # 根据数据量调整
cutoff_len = 2048        # 序列长度
warmup_steps = 100       # Warmup 步数
weight_decay = 0.01      # 权重衰减
```

## 性能优化

### 1. 调整 Batch Size

```bash
# 如果显存充足，可以增加 micro_batch_size
--micro_batch_size 2

# 相应减少 gradient_accumulation_steps
# 保持 effective_batch_size = micro_batch_size × gradient_accumulation_steps × world_size
```

### 2. 序列长度

```bash
# 根据实际数据分布调整
--cutoff_len 1024  # 短文本任务
--cutoff_len 4096  # 长文本任务
```

### 3. 数据加载

```python
# 在 TrainingArguments 中
dataloader_num_workers=4  # 多进程加载
dataloader_pin_memory=False  # NPU 建议关闭
```

## 监控和调试

### 查看训练日志

```bash
# 实时查看
tail -f ./trained_models/qwen32b-lora-sft/trainer_log.txt

# 查看 DeepSpeed 日志
cat ./trained_models/qwen32b-lora-sft/deepspeed_*.log
```

### 常见问题

#### 1. OOM (Out of Memory)

**解决方案**:
- 减小 `micro_batch_size` 到 1
- 减小 `cutoff_len`
- 启用更激进的 CPU offload

#### 2. 通信超时

**解决方案**:
```bash
export NCCL_TIMEOUT=1800  # 增加超时时间
export HCCL_CONNECT_TIMEOUT=1800
```

#### 3. 节点间通信失败

**检查**:
- 确保两个节点可以互相 ping 通
- 确保防火墙开放端口 29500
- 确保 MASTER_ADDR 设置正确

#### 4. 课程学习未生效

**检查**:
- 确认数据集包含 `difficulty` 字段
- 检查日志中是否有 "Warning: 'difficulty' field not found"
- 确认 `--use_curriculum True` 参数已设置

#### 5. 数据集 difficulty 分布不均

**解决方案**:
- 课程学习会自动处理重复采样
- 如果某个难度级别样本不足，会重复采样以达到目标比例
- 建议各难度级别至少有 1000+ 样本

## 模型保存和加载

### 保存

训练完成后，模型保存在 `output_dir`:
```
output_dir/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer_config.json
└── ...
```

### 加载

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-32B")
model = PeftModel.from_pretrained(base_model, "./trained_models/qwen32b-lora-sft")
tokenizer = AutoTokenizer.from_pretrained("./trained_models/qwen32b-lora-sft")
```

## 性能基准

### 预期训练速度 (16×910B)

| Batch Size | Seq Len | Speed (samples/s) | 显存占用 |
|------------|---------|-------------------|----------|
| 128        | 2048    | ~50-80            | ~50GB    |
| 64         | 4096    | ~30-50            | ~55GB    |

*实际速度取决于数据复杂度和网络带宽*

## 参考资料

- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [Qwen 模型文档](https://github.com/QwenLM/Qwen)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [昇腾 NPU 文档](https://www.hiascend.com/)

## 技术支持

如遇问题，请检查:
1. CANN 版本是否为 8.2.rc1
2. DeepSpeed 是否正确安装
3. 多节点网络连接是否正常
4. 日志文件中的详细错误信息
