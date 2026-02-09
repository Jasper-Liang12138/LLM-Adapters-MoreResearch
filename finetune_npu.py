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
from datasets import load_dataset, concatenate_datasets
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

    print(f"Process rank {local_rank}: Model loaded successfully on CPU, moving to {device}...")

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
        注意：输入的数据集应该已经 tokenized
        使用简单的采样策略，避免 interleave_datasets 的复杂性
        """
        import random

        probs = get_curriculum_probs(progress)
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

        print(f"✅ Curriculum dataset built: {len(mixed_ds)} samples")

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

    # 关键：先应用 PEFT，再移动到设备
    # 这样可以确保 LoRA 参数正确初始化并启用梯度
    print(f"Process rank {local_rank}: Moving PEFT model to {device}...")
    model = model.to(device)

    # 清理缓存
    torch.npu.empty_cache()

    # 确保模型在训练模式
    model.train()

    # 显式确保 LoRA 参数启用梯度
    for param in model.parameters():
        if param.requires_grad:
            # 确保梯度已启用的参数保持启用状态
            param.requires_grad = True

    print(f"Process rank {local_rank}: PEFT model ready on {device}")

    # --- 启用梯度检查点以节省显存 ---
    # 注意：在 NPU 上，梯度检查点可能与 PEFT 不兼容，暂时禁用
    use_gradient_checkpointing = False  # 强制禁用，避免梯度问题
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
        print("ℹ️  Gradient checkpointing disabled (recommended for NPU + PEFT compatibility)")

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

            # 关键优化：先 tokenize 各个子数据集，再 interleave
            # 这样可以显示进度条，而且只需要 tokenize 一次
            print(f"🔄 Pre-tokenizing datasets (this will be done once)...")
            explain_ds = explain_ds.map(generate_and_tokenize_prompt, batched=False, num_proc=1, desc="Tokenizing Explain")
            reasoning_ds = reasoning_ds.map(generate_and_tokenize_prompt, batched=False, num_proc=1, desc="Tokenizing Reasoning")
            topology_ds = topology_ds.map(generate_and_tokenize_prompt, batched=False, num_proc=1, desc="Tokenizing Topology")
            print(f"✅ Pre-tokenization complete!\n")
    else:
        # 非 curriculum 模式：直接处理数据
        print(f"🔄 Tokenizing dataset (this may take a moment)...")
        train_data = data["train"].shuffle().map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=1,
            desc="Tokenizing"
        )
        print(f"✅ Tokenization complete!")

    gradient_accumulation_steps = (batch_size // micro_batch_size) // world_size

    # 根据是否使用 curriculum learning 决定初始训练数据
    if use_curriculum:
        # Curriculum learning: 先用第一个 epoch 的数据（progress=0）
        initial_progress = 0.0
        initial_mixed_ds = build_curriculum_dataset(explain_ds, reasoning_ds, topology_ds, initial_progress)
        # 数据已经 tokenized，只需要 shuffle
        initial_train_data = initial_mixed_ds.shuffle()
    else:
        print(f"🔄 Tokenizing dataset (this may take a moment)...")
        initial_train_data = train_data.map(
            generate_and_tokenize_prompt,
            batched=False,
            num_proc=1,
            desc="Tokenizing"
        )
        print(f"✅ Tokenization complete! Ready to train.")

    # ### 修改点 5: 核心修复 - 开启 bf16=True
    # 这会告诉 Trainer 不要使用 GradScaler，因为 BF16 不需要缩放。
    # 彻底解决 "Loss scaler reducing loss scale to 0.0" 问题

    # 自定义回调：打印训练进度和指标
    class ProgressCallback(transformers.TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            """每次日志记录时调用"""
            if logs and state.is_local_process_zero:
                # 只在主进程打印
                step = state.global_step
                epoch = state.epoch if state.epoch is not None else 0

                # 构建日志信息
                log_str = f"[Step {step} | Epoch {epoch:.2f}]"

                if "loss" in logs:
                    log_str += f" Loss: {logs['loss']:.4f}"
                if "learning_rate" in logs:
                    log_str += f" | LR: {logs['learning_rate']:.2e}"
                if "grad_norm" in logs:
                    log_str += f" | Grad Norm: {logs['grad_norm']:.4f}"

                # 打印到控制台
                if "loss" in logs:  # 只在有 loss 时打印，避免重复
                    print(log_str)

        def on_epoch_end(self, _args, state, _control, **_kwargs):
            """每个 epoch 结束时调用"""
            if state.is_local_process_zero:
                print(f"\n{'='*60}")
                print(f"✅ Epoch {int(state.epoch)} completed!")
                print(f"   Total steps: {state.global_step}")
                print(f"{'='*60}\n")

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

            # --- 日志和进度显示配置 ---
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,  # 显示第一步的指标
            disable_tqdm=False,  # 启用进度条
            # ----------------

            dataloader_pin_memory=False,
            save_strategy="steps",
            save_steps=save_step,
            output_dir=output_dir,
            ddp_find_unused_parameters=False,
            report_to="none",

        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
        callbacks=[ProgressCallback()],  # 添加自定义回调
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
                # 注意：传入 None 而不是 True，避免尝试加载 checkpoint
                trainer.train(resume_from_checkpoint=None)

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