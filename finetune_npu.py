import os
import sys

# ### 修改点 1: 在导入 torch 之前设置环境变量
# 强制 NPU 保持原图数据类型，禁止私自转 FP16，这是解决溢出的关键
os.environ["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
# 禁止某些可能导致 Inner Error 的融合算子
os.environ["LCCL_DETERMINISTIC"] = "1"
os.environ["HCC_DETERMINISTIC"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
# 关键：某些环境下禁用这个可以解决 RMSNorm 导致的 Inner Error
os.environ["ACL_PRECISION_MODE"] = "must_keep_origin_dtype" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Optional, Union
import json
import fire
import torch
import torch_npu  # 核心：昇腾必需
from torch_npu.contrib import transfer_to_npu
import time  # 用于进程间延迟加载

# ### 修改点 2: NPU 设置
# 关闭 JIT 编译（Qwen2.5 在 NPU 上 JIT 有时会不稳定）
torch_npu.npu.set_compile_mode(jit_compile=False)
# 再次通过 API 确保精度模式（双重保险）
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

import transformers
from datasets import load_dataset, Dataset, interleave_datasets
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (
    LoraConfig, AdaLoraConfig, BottleneckConfig, PrefixTuningConfig,
    get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 工具函数保持不变 ---
def safe_int(value, default=0):
    try: return int(value)
    except: return default
def safe_float(value, default=0.0):
    try: return float(value)
    except: return default
def safe_bool(value, default=False):
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.lower() in ('true', '1', 'yes', 'y', 't')
    return bool(value)
def safe_list(value, default=None):
    if default is None: default = []
    if value is None: return default
    if isinstance(value, list): return value
    return default
def safe_str(value, default=""):
    if value is None: return default
    return str(value)

def train(
        base_model: str = "",
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-qwen",
        adapter_name: str = "lora",
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,
        val_set_size: int = 0,
        eval_step: int = 50,
        save_step: int = 200,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",
        wandb_log_model: str = "",
        resume_from_checkpoint: str = None,
        use_tf_grpo: bool = False,
        grpo_group_size: int = 4,
        grpo_max_experiences: int = 50,
        grpo_data_limit: int = -1,
        # === Curriculum Learning 参数 ===
        use_curriculum: bool = True,  # 是否启用课程学习
        curriculum_seed: int = 42,  # 数据混合的随机种子
        # === 内存优化参数 ===
        use_gradient_checkpointing: bool = True,  # 梯度检查点，节省显存（推荐开启）
):
    print(f"Finetuning Qwen2.5 on Ascend NPU with Full BF16 Precision...")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 直接设置当前进程使用的设备
    torch.npu.set_device(local_rank)
    device = f'npu:{local_rank}'

    # 清理当前 NPU 的缓存
    torch.npu.empty_cache()

    # 进程间错开加载，避免同时读取模型文件导致 I/O 瓶颈
    if world_size > 1 and local_rank > 0:
        time.sleep(local_rank * 3)

    print(f"Process rank {local_rank}/{world_size}: Loading model to {device}...")

    # 简化加载：直接加载到指定设备，不使用 device_map 和 max_memory
    # 这些参数在分布式训练时会导致设备分配混乱
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 手动将模型移动到指定设备
    model = model.to(device)

    # 清理缓存
    torch.npu.empty_cache()

    print(f"Process rank {local_rank}: Model loaded successfully on {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def generate_and_tokenize_prompt(data_point):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["instruction"] + ("\n" + data_point["input"] if data_point.get("input") else "")},
            {"role": "assistant", "content": data_point["output"]}
        ]
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, truncation=True, max_length=cutoff_len)
        labels = list(full_tokens)
        if not train_on_inputs:
            user_tokens = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            user_len = len(user_tokens)
            labels = [-100] * user_len + labels[user_len:]
            if len(labels) > len(full_tokens): labels = labels[:len(full_tokens)]
        return {"input_ids": full_tokens, "attention_mask": [1] * len(full_tokens), "labels": labels}

    # === Curriculum Learning Helper Functions ===
    def get_curriculum_probs(progress):
        """
        根据训练进度返回不同难度数据的采样概率
        progress: 0.0 到 1.0 之间的训练进度
        返回: [explain_prob, reasoning_prob, topology_prob]
        """
        if progress < 0.3:
            # 早期阶段：主要学习解释性内容
            return [0.8, 0.2, 0.0]
        elif progress < 0.7:
            # 中期阶段：平衡解释和推理
            return [0.4, 0.4, 0.2]
        else:
            # 后期阶段：更多推理和拓扑内容
            return [0.25, 0.35, 0.4]

    def build_curriculum_dataset(explain_ds, reasoning_ds, topology_ds, progress):
        """
        根据训练进度动态构建混合数据集
        """
        probs = get_curriculum_probs(progress)
        print(f"📚 Curriculum Progress: {progress:.2%} | Sampling Probs: Explain={probs[0]:.2f}, Reasoning={probs[1]:.2f}, Topology={probs[2]:.2f}")

        # 使用 interleave_datasets 按概率混合数据
        mixed_ds = interleave_datasets(
            [explain_ds, reasoning_ds, topology_ds],
            probabilities=probs,
            seed=curriculum_seed,
            stopping_strategy="all_exhausted"
        )

        return mixed_ds

    target_modules = safe_list(target_modules, ["q_proj", "k_proj", "v_proj", "o_proj"])

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules, 
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "adalora":
        config = AdaLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            # ===== AdaLoRA 关键参数 =====
            init_r=lora_r,
            target_r=lora_r,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            deltaT=10,
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config)

    # PEFT 包装后清理缓存
    torch.npu.empty_cache()
    # prefix-tuning 需要确保在正确的设备上（其他适配器已经在正确设备）
    if adapter_name == "prefix-tuning":
        model = model.to(device)

    # --- 启用梯度检查点以节省显存 ---
    if use_gradient_checkpointing:
        # 必须在 PEFT 包装后启用，这样 LoRA 层也能受益
        # NPU 兼容性检查：gradient checkpointing 在 NPU 上通常可用，但需要测试
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                print(f"✅ Gradient checkpointing enabled (saves ~30-50% memory)")
                print(f"   Note: If you encounter errors, try --use_gradient_checkpointing False")
            elif hasattr(model, "gradient_checkpointing"):
                # 某些模型使用不同的属性名
                model.gradient_checkpointing = True
                print(f"✅ Gradient checkpointing enabled via gradient_checkpointing attribute")
            else:
                print("⚠️  Warning: Model does not support gradient checkpointing, disabling...")
                use_gradient_checkpointing = False
        except Exception as e:
            print(f"⚠️  Warning: Failed to enable gradient checkpointing on NPU: {e}")
            print(f"   Disabling gradient checkpointing. If memory issues persist, try reducing batch_size.")
            use_gradient_checkpointing = False
    else:
        print("ℹ️  Gradient checkpointing disabled")

    # 打印可训练参数，确认 LoRA 挂载正确
    model.print_trainable_parameters()

    # ### 修改点 4: 移除手动的 model.bfloat16()，依赖 Trainer 的参数控制
    # 因为 Trainer 会根据 args.bf16 自动处理，手动转换有时会扰乱 Trainer 的状态
    
    data = load_dataset("json", data_files=data_path) if data_path.endswith(".json") else load_dataset(data_path)

    # === Curriculum Learning Data Preparation ===
    explain_ds = None
    reasoning_ds = None
    topology_ds = None

    if use_curriculum:
        print("\n" + "="*50)
        print("📚 Starting Curriculum Learning Setup")
        print("Filtering dataset by difficulty levels...")
        print("="*50 + "\n")

        full_ds = data["train"]

        # 检查数据集中是否有 difficulty 字段
        if "difficulty" not in full_ds.column_names:
            print("⚠️  Warning: 'difficulty' field not found in dataset. Disabling curriculum learning.")
            use_curriculum = False
            train_data = full_ds.shuffle().map(generate_and_tokenize_prompt)
        else:
            # 按难度过滤数据集
            explain_ds = full_ds.filter(lambda x: x.get("difficulty") == "explain")
            reasoning_ds = full_ds.filter(lambda x: x.get("difficulty") == "reasoning")
            topology_ds = full_ds.filter(lambda x: x.get("difficulty") == "topology")

            # 如果没有 topology 数据，使用 reasoning 的一部分
            if len(topology_ds) == 0:
                print("⚠️  No 'topology' difficulty data found. Using reasoning data for topology stage.")
                topology_ds = reasoning_ds

            print(f"✅ Dataset split by difficulty:")
            print(f"   - Explain: {len(explain_ds)} samples")
            print(f"   - Reasoning: {len(reasoning_ds)} samples")
            print(f"   - Topology: {len(topology_ds)} samples")
            print(f"   - Total: {len(full_ds)} samples\n")
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    gradient_accumulation_steps = (batch_size // micro_batch_size) // world_size

    # 根据是否使用 curriculum learning 决定初始训练数据
    if use_curriculum:
        # Curriculum learning: 先用第一个 epoch 的数据（progress=0）
        initial_progress = 0.0
        initial_mixed_ds = build_curriculum_dataset(explain_ds, reasoning_ds, topology_ds, initial_progress)
        initial_train_data = initial_mixed_ds.shuffle().map(generate_and_tokenize_prompt)
    else:
        initial_train_data = train_data

    # ### 修改点 5: 核心修复 - 开启 bf16=True
    # 这会告诉 Trainer 不要使用 GradScaler，因为 BF16 不需要缩放。
    # 彻底解决 "Loss scaler reducing loss scale to 0.0" 问题

    trainer = transformers.Trainer(
        model=model,
        train_dataset=initial_train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=1 if use_curriculum else num_epochs,  # Curriculum: 每次训练1个epoch
            learning_rate=learning_rate,

            # --- 关键修改 ---
            bf16=True,      # 必须开启！这会禁用 FP16 GradScaler
            fp16=False,     # 确保关闭 FP16
            optim="adamw_torch",
            gradient_checkpointing=use_gradient_checkpointing,  # 梯度检查点，节省显存
            # ----------------

            dataloader_pin_memory=False,
            logging_steps=10,
            save_strategy="steps",
            save_steps=save_step,
            output_dir=output_dir,
            ddp_find_unused_parameters=False,
            report_to="none",

        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
    )

    # 禁用 KV cache（训练时不需要）
    model.config.use_cache = False

    # 清理内存
    torch.npu.empty_cache()

    # =================================================================
    # Training Loop with Curriculum Learning Support
    # =================================================================
    if use_curriculum:
        print("\n" + "="*50)
        print("🎓 Starting Curriculum Learning Training")
        print(f"Total Epochs: {num_epochs}")
        print("="*50 + "\n")

        for epoch in range(num_epochs):
            # 计算当前训练进度
            progress = epoch / num_epochs

            print(f"\n{'='*50}")
            print(f"📖 Epoch {epoch + 1}/{num_epochs} (Progress: {progress:.2%})")
            print(f"{'='*50}\n")

            # 动态构建当前 epoch 的数据集
            mixed_ds = build_curriculum_dataset(explain_ds, reasoning_ds, topology_ds, progress)
            current_train_data = mixed_ds.shuffle().map(generate_and_tokenize_prompt)

            # 更新 trainer 的训练数据集
            trainer.train_dataset = current_train_data

            # 训练当前 epoch
            if epoch == 0:
                # 第一个 epoch，可能需要从 checkpoint 恢复
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                # 后续 epoch，从上一个 epoch 的结果继续
                trainer.train(resume_from_checkpoint=True)

            print(f"\n✅ Epoch {epoch + 1}/{num_epochs} completed!\n")

        print("\n" + "="*50)
        print("🎉 Curriculum Learning Training Completed!")
        print("="*50 + "\n")
    else:
        # 标准训练（不使用 curriculum learning）
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)