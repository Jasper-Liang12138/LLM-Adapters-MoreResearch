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
from datasets import load_dataset, concatenate_datasets
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
        micro_batch_size: int = 1,  # ZeRO-3 建议使用更小的 micro_batch_size
        num_epochs: int = 3,
        learning_rate: float = 2e-5,  # 大模型建议使用更小的学习率
        cutoff_len: int = 2048,
        val_set_size: int = 0,
        save_step: int = 500,
        lora_r: int = 64,  # 32B 模型建议使用更大的 rank
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        train_on_inputs: bool = False,  # 通常不训练 instruction 部分
        resume_from_checkpoint: str = None,
        deepspeed_config: str = None,  # DeepSpeed 配置文件路径
        # === Curriculum Learning 参数 ===
        use_curriculum: bool = True,  # 是否启用课程学习
        curriculum_seed: int = 42,  # 数据混合的随机种子
        local_rank: int = -1,  # DeepSpeed 会自动设置
):
    """
    使用 DeepSpeed ZeRO-3 训练 Qwen-32B
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
        attn_implementation="eager",  # flash_attention_2 在 NPU 上可能不稳定
        use_cache=False,  # 训练时必须禁用
    )

    print(f"✅ Model loaded on rank {rank}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 左填充，适合生成任务

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
        注意：输入的数据集应该已经 tokenized
        使用简单的采样策略，避免 interleave_datasets 的复杂性
        """
        import random

        probs = get_curriculum_probs(progress)
        if rank == 0:  # 只在主进程打印
            print(f"📚 Curriculum Progress: {progress:.2%} | Sampling Probs: Explain={probs[0]:.2f}, Reasoning={probs[1]:.2f}, Topology={probs[2]:.2f}")

        # 简单策略：按概率计算每个数据集应该取多少样本
        total_samples = len(explain_ds) + len(reasoning_ds) + len(topology_ds)

        # 计算每个数据集的目标样本数
        n_explain = int(total_samples * probs[0])
        n_reasoning = int(total_samples * probs[1])
        n_topology = int(total_samples * probs[2])

        # 确保总数正确（处理舍入误差）
        diff = total_samples - (n_explain + n_reasoning + n_topology)
        if diff > 0:
            n_explain += diff

        if rank == 0:
            print(f"🔄 Building curriculum dataset: {n_explain} explain + {n_reasoning} reasoning + {n_topology} topology = {n_explain + n_reasoning + n_topology} samples")

        # 从每个数据集中随机采样（允许重复采样以达到目标数量）
        random.seed(curriculum_seed + int(progress * 1000))  # 每个 epoch 不同的种子

        sampled_datasets = []
        if n_explain > 0:
            # 如果需要的样本数超过数据集大小，允许重复采样
            if n_explain <= len(explain_ds):
                indices = random.sample(range(len(explain_ds)), n_explain)
            else:
                # 重复采样：先全部取，然后随机补充
                indices = list(range(len(explain_ds)))
                indices += random.choices(range(len(explain_ds)), k=n_explain - len(explain_ds))
            sampled_datasets.append(explain_ds.select(indices))
        if n_reasoning > 0:
            if n_reasoning <= len(reasoning_ds):
                indices = random.sample(range(len(reasoning_ds)), n_reasoning)
            else:
                indices = list(range(len(reasoning_ds)))
                indices += random.choices(range(len(reasoning_ds)), k=n_reasoning - len(reasoning_ds))
            sampled_datasets.append(reasoning_ds.select(indices))
        if n_topology > 0:
            if n_topology <= len(topology_ds):
                indices = random.sample(range(len(topology_ds)), n_topology)
            else:
                indices = list(range(len(topology_ds)))
                indices += random.choices(range(len(topology_ds)), k=n_topology - len(topology_ds))
            sampled_datasets.append(topology_ds.select(indices))

        # 合并所有采样的数据集
        mixed_ds = concatenate_datasets(sampled_datasets)

        if rank == 0:
            print(f"✅ Curriculum dataset built: {len(mixed_ds)} samples")

        return mixed_ds

    # LoRA 配置
    target_modules = safe_list(target_modules, [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # Qwen 的 MLP 层
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
    print(f"📚 Loading dataset from {data_path}")
    data = load_dataset("json", data_files=data_path) if data_path.endswith(".json") else load_dataset(data_path)

    # === Curriculum Learning Data Preparation ===
    explain_ds = None
    reasoning_ds = None
    topology_ds = None

    if use_curriculum:
        if rank == 0:
            print("\n" + "="*50)
            print("📚 Starting Curriculum Learning Setup")
            print("Filtering dataset by difficulty levels...")
            print("="*50 + "\n")

        full_ds = data["train"]

        # 检查数据集中是否有 difficulty 字段
        if "difficulty" not in full_ds.column_names:
            if rank == 0:
                print("⚠️  Warning: 'difficulty' field not found in dataset. Disabling curriculum learning.")
            use_curriculum = False
            train_data = full_ds.shuffle().map(
                generate_and_tokenize_prompt,
                batched=False,
                num_proc=4,
                desc="Tokenizing"
            )
        else:
            # 按难度过滤数据集
            explain_ds = full_ds.filter(lambda x: x.get("difficulty") == "explain")
            reasoning_ds = full_ds.filter(lambda x: x.get("difficulty") == "reasoning")
            topology_ds = full_ds.filter(lambda x: x.get("difficulty") == "topology")

            # 如果没有 topology 数据，使用 reasoning 的一部分
            if len(topology_ds) == 0:
                if rank == 0:
                    print("⚠️  No 'topology' difficulty data found. Using reasoning data for topology stage.")
                topology_ds = reasoning_ds

            if rank == 0:
                print(f"✅ Dataset split by difficulty:")
                print(f"   - Explain: {len(explain_ds)} samples")
                print(f"   - Reasoning: {len(reasoning_ds)} samples")
                print(f"   - Topology: {len(topology_ds)} samples")
                print(f"   - Total: {len(full_ds)} samples\n")

            # 关键优化：先 tokenize 各个子数据集，再 interleave
            # 这样可以显示进度条，而且只需要 tokenize 一次
            if rank == 0:
                print(f"🔄 Pre-tokenizing datasets (this will be done once)...")
            explain_ds = explain_ds.map(generate_and_tokenize_prompt, batched=False, num_proc=4, desc="Tokenizing Explain")
            reasoning_ds = reasoning_ds.map(generate_and_tokenize_prompt, batched=False, num_proc=4, desc="Tokenizing Reasoning")
            topology_ds = topology_ds.map(generate_and_tokenize_prompt, batched=False, num_proc=4, desc="Tokenizing Topology")
            if rank == 0:
                print(f"✅ Pre-tokenization complete!\n")
    else:
        # 非 curriculum 模式：直接处理数据
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

    # 根据是否使用 curriculum learning 决定初始训练数据
    if use_curriculum:
        # Curriculum learning: 先用第一个 epoch 的数据（progress=0）
        initial_progress = 0.0
        initial_mixed_ds = build_curriculum_dataset(explain_ds, reasoning_ds, topology_ds, initial_progress)
        # 数据已经 tokenized，只需要 shuffle
        initial_train_data = initial_mixed_ds.shuffle()
    else:
        initial_train_data = train_data

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
        print(f"💾 DeepSpeed config saved to {ds_config_path}")
        deepspeed_config = ds_config_path

    # 训练参数
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1 if use_curriculum else num_epochs,  # Curriculum: 每次训练1个epoch
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
        deepspeed=deepspeed_config,  # 启用 DeepSpeed
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
        train_dataset=initial_train_data,
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

    # =================================================================
    # Training Loop with Curriculum Learning Support
    # =================================================================
    if use_curriculum:
        if rank == 0:
            print("\n" + "="*50)
            print("🎓 Starting Curriculum Learning Training")
            print(f"Total Epochs: {num_epochs}")
            print("="*50 + "\n")

        for epoch in range(num_epochs):
            # 计算当前训练进度
            progress = epoch / num_epochs

            if rank == 0:
                print(f"\n{'='*50}")
                print(f"📖 Epoch {epoch + 1}/{num_epochs} (Progress: {progress:.2%})")
                print(f"{'='*50}\n")

            # 动态构建当前 epoch 的数据集
            mixed_ds = build_curriculum_dataset(explain_ds, reasoning_ds, topology_ds, progress)
            # 数据已经 tokenized，只需要 shuffle
            current_train_data = mixed_ds.shuffle()

            # 更新 trainer 的训练数据集
            trainer.train_dataset = current_train_data

            # 训练当前 epoch
            if epoch == 0:
                # 第一个 epoch，可能需要从 checkpoint 恢复
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                # 后续 epoch：不从 checkpoint 恢复，直接继续训练
                # 模型已经在内存中并且已训练，不需要重新加载
                trainer.train(resume_from_checkpoint=None)

            if rank == 0:
                print(f"\n✅ Epoch {epoch + 1}/{num_epochs} completed!\n")

        if rank == 0:
            print("\n" + "="*50)
            print("🎉 Curriculum Learning Training Completed!")
            print("="*50 + "\n")
    else:
        # 标准训练（不使用 curriculum learning）
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型（只在 rank 0 保存）
    if rank == 0:
        print(f"\n💾 Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Training completed!")

if __name__ == "__main__":
    fire.Fire(train)
