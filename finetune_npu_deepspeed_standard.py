import os
import sys

# ### NPU 环境变量设置
os.environ["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
os.environ["LCCL_DETERMINISTIC"] = "1"
os.environ["HCC_DETERMINISTIC"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Optional, Union
import json
import fire
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import time

# NPU 设置
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

import transformers
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (
    LoraConfig, get_peft_model
)
from transformers import AutoModelForCausalLM, AutoTokenizer

def safe_list(value, default=None):
    if default is None: default = []
    if value is None: return default
    if isinstance(value, list): return value
    return default

def train(
        base_model: str = "",
        data_path: str = "",
        output_dir: str = "./qwen32b-lora",
        adapter_name: str = "lora",
        batch_size: int = 128,
        micro_batch_size: int = 1,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        cutoff_len: int = 2048,
        val_set_size: int = 0,
        save_step: int = 500,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        train_on_inputs: bool = False,
        resume_from_checkpoint: str = None,
        deepspeed_config: str = None,
        local_rank: int = -1,
):
    """
    使用 DeepSpeed ZeRO-3 训练 Qwen-32B (标准微调版本)
    支持多节点分布式训练
    """
    print(f"🚀 Finetuning Qwen-32B on Ascend NPU with DeepSpeed ZeRO-3...")

    # DeepSpeed 会自动设置这些环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", "0"))

    print(f"📊 Distributed Info: Rank {rank}/{world_size}, Local Rank {local_rank}")

    # 设置当前进程使用的 NPU
    torch.npu.set_device(local_rank)

    # 清理缓存
    torch.npu.empty_cache()

    # 多节点训练时，错开模型加载时间
    if world_size > 1 and rank > 0:
        time.sleep(rank * 2)

    print(f"💾 Loading model: {base_model}")

    # 使用 DeepSpeed 时，模型初始化在 CPU 上，DeepSpeed 会处理分片
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False,
    )

    print(f"✅ Model loaded on rank {rank}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def generate_and_tokenize_prompt(data_point):
        """数据预处理函数"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data_point["instruction"] + ("\n" + data_point["input"] if data_point.get("input") else "")},
            {"role": "assistant", "content": data_point["output"]}
        ]

        full_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            truncation=True,
            max_length=cutoff_len
        )
        labels = list(full_tokens)

        if not train_on_inputs:
            # 只计算 assistant 部分的 loss
            user_tokens = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True
            )
            user_len = len(user_tokens)
            labels = [-100] * user_len + labels[user_len:]
            if len(labels) > len(full_tokens):
                labels = labels[:len(full_tokens)]

        return {
            "input_ids": full_tokens,
            "attention_mask": [1] * len(full_tokens),
            "labels": labels
        }

    # LoRA 配置
    target_modules = safe_list(target_modules, [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 加载数据集
    if rank == 0:
        print(f"📚 Loading dataset from {data_path}")
    data = load_dataset("json", data_files=data_path) if data_path.endswith(".json") else load_dataset(data_path)

    # 标准数据处理
    if rank == 0:
        print(f"🔄 Tokenizing dataset...")
    train_data = data["train"].shuffle().map(
        generate_and_tokenize_prompt,
        batched=False,
        num_proc=4,
        desc="Tokenizing"
    )
    if rank == 0:
        print(f"✅ Tokenization complete! Total samples: {len(train_data)}")

    # 计算梯度累积步数
    gradient_accumulation_steps = batch_size // (micro_batch_size * world_size)

    if rank == 0:
        print(f"⚙️  Training Configuration:")
        print(f"   - World Size: {world_size}")
        print(f"   - Micro Batch Size: {micro_batch_size}")
        print(f"   - Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"   - Effective Batch Size: {micro_batch_size * gradient_accumulation_steps * world_size}")

    # DeepSpeed 配置
    if deepspeed_config is None:
        # 如果没有提供配置文件，使用默认配置
        deepspeed_config = {
            "train_batch_size": batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "bf16": {
                "enabled": True
            },
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }

        # 保存配置到文件
        ds_config_path = os.path.join(output_dir, "ds_config.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(ds_config_path, "w") as f:
            json.dump(deepspeed_config, f, indent=2)
        if rank == 0:
            print(f"💾 DeepSpeed config saved to {ds_config_path}")
        deepspeed_config = ds_config_path

    # 训练参数
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        fp16=False,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_step,
        save_total_limit=3,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_config,
        report_to="none",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    # Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )

    # 开始训练
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🎯 Starting Training...")
        print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型（只在 rank 0 保存）
    if rank == 0:
        print(f"\n💾 Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Training completed!")

if __name__ == "__main__":
    fire.Fire(train)
