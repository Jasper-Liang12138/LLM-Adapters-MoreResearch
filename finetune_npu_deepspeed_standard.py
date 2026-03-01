import sys
try:
    import lzma
except ModuleNotFoundError:
    from backports import lzma
    sys.modules["lzma"] = lzma

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
# 注意：不导入 transfer_to_npu，避免其劫持 .to() 方法干扰 DeepSpeed ZeRO-3 分片
# from torch_npu.contrib import transfer_to_npu
import time

# NPU 设置
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

# 手动设置 hccl 后端（原本由 transfer_to_npu 完成）
import torch.distributed
_original_init_process_group = torch.distributed.init_process_group
def _patched_init_process_group(*args, **kwargs):
    if "backend" not in kwargs and (not args or args[0] not in ("hccl", "nccl", "gloo", "mpi")):
        kwargs["backend"] = "hccl"
    elif args and args[0] == "nccl":
        args = ("hccl",) + args[1:]
    return _original_init_process_group(*args, **kwargs)
torch.distributed.init_process_group = _patched_init_process_group

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

    # 提前初始化分布式进程组（deepspeed.zero.Init 需要）
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hccl")

    # 计算梯度累积步数（DeepSpeed zero.Init 需要提前知道）
    gradient_accumulation_steps = batch_size // (micro_batch_size * world_size)

    # ---- 提前解析/生成 DeepSpeed 配置文件路径 ----
    # deepspeed.zero.Init 必须在 from_pretrained 之前拿到配置 dict
    if deepspeed_config is None:
        ds_config_dict = {
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
                "overlap_comm": False,
                "contiguous_gradients": True,
                "sub_group_size": 1e8,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 2e7,
                "stage3_max_reuse_distance": 0,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "bf16": {
                "enabled": True
            },
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
        os.makedirs(output_dir, exist_ok=True)
        ds_config_path = os.path.join(output_dir, "ds_config.json")
        with open(ds_config_path, "w") as f:
            json.dump(ds_config_dict, f, indent=2)
        deepspeed_config = ds_config_path
    else:
        with open(deepspeed_config) as f:
            ds_config_dict = json.load(f)

    if rank == 0:
        print(f"💾 DeepSpeed config: {deepspeed_config}")

    # 多节点训练时，错开模型加载时间
    if world_size > 1 and rank > 0:
        time.sleep(rank * 2)

    print(f"💾 Loading model: {base_model}")

    # Patch RoPE forward 在 from_pretrained 之前打好补丁
    import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
    _original_rope_forward = qwen2_modeling.Qwen2RotaryEmbedding.forward
    def _patched_rope_forward(self, x, position_ids):
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        return _original_rope_forward(self, x, position_ids)
    qwen2_modeling.Qwen2RotaryEmbedding.forward = _patched_rope_forward

    # 使用 deepspeed.zero.Init 上下文加载模型
    # 这使参数直接以分片形式存在于 CPU，offload_param 才真正生效
    # remote_device="cpu" 确保初始化时参数在 CPU 而非 meta/NPU
    # deepspeed.zero.Init 不支持 "auto"，需要传入具体数值
    import deepspeed
    _zero_init_cfg = {
        "train_batch_size": micro_batch_size * gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
        },
    }
    with deepspeed.zero.Init(
        remote_device="cpu",
        pin_memory=True,
        config_dict_or_path=_zero_init_cfg,
        dtype=torch.bfloat16,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            use_cache=False,
            local_files_only=True,
        )

    print(f"✅ Model loaded on rank {rank}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
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
        desc="Tokenizing",
        remove_columns=data["train"].column_names,
    )
    if rank == 0:
        print(f"✅ Tokenization complete! Total samples: {len(train_data)}")

    if rank == 0:
        print(f"⚙️  Training Configuration:")
        print(f"   - World Size: {world_size}")
        print(f"   - Micro Batch Size: {micro_batch_size}")
        print(f"   - Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"   - Effective Batch Size: {micro_batch_size * gradient_accumulation_steps * world_size}")

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
