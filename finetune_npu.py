import os
import sys

# ### 修改点 1: 在导入 torch 之前设置环境变量
# 强制 NPU 保持原图数据类型，禁止私自转 FP16，这是解决溢出的关键
os.environ["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
# 优化显存分配，防止碎片化
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:128"
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

# ### 修改点 2: NPU 设置
# 关闭 JIT 编译（Qwen2.5 在 NPU 上 JIT 有时会不稳定）
torch_npu.npu.set_compile_mode(jit_compile=False)
# 再次通过 API 确保精度模式（双重保险）
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

import transformers
from datasets import load_dataset, Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (
    LoraConfig, AdaLoraConfig, QLoRAConfig, BottleneckConfig, PrefixTuningConfig,
    get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict,
    prepare_model_for_int8_training, prepare_model_for_kbit_training
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
        load_8bit: bool = False,
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,
        val_set_size: int = 0,
        use_gradient_checkpointing: bool = False,
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
):
    print(f"Finetuning Qwen2.5 on Ascend NPU with Full BF16 Precision...")
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    device_map = {"": local_rank}

    # ### 修改点 3: 显式指定 torch_dtype 为 bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16, 
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager" 
    )

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

    if load_8bit:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    elif adapter_name == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    
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
    elif adapter_name == "qlora":
        config = QLoRAConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
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
    if adapter_name == "prefix-tuning":
        model.to('cuda')
    model = get_peft_model(model, config)

    # --- NPU 稳定性补丁：修复 RMSNorm 导致的 Inner Error ---
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    for name, module in model.named_modules():
        if isinstance(module, Qwen2RMSNorm):
            # 强制这些层在正向传播时转换类型，规避 NPU 算子 bug
            module.float() 
    # -----------------------------------------------------

    # 打印可训练参数，确认 LoRA 挂载正确
    model.print_trainable_parameters()

    # ### 修改点 4: 移除手动的 model.bfloat16()，依赖 Trainer 的参数控制
    # 因为 Trainer 会根据 args.bf16 自动处理，手动转换有时会扰乱 Trainer 的状态
    
    data = load_dataset("json", data_files=data_path) if data_path.endswith(".json") else load_dataset(data_path)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    gradient_accumulation_steps = (batch_size // micro_batch_size) // world_size
    
    # ### 修改点 5: 核心修复 - 开启 bf16=True
    # 这会告诉 Trainer 不要使用 GradScaler，因为 BF16 不需要缩放。
    # 彻底解决 "Loss scaler reducing loss scale to 0.0" 问题
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            
            # --- 关键修改 ---
            bf16=True,      # 必须开启！这会禁用 FP16 GradScaler
            fp16=False,     # 确保关闭 FP16
            optim="adamw_torch",
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
    # 防止 Qwen 的 token_type_ids 导致 NaN (Qwen 不需要这个，但有时会被自动加上)
    if hasattr(model, "config"):
        model.config.use_cache = False
    
    # 强制清理一下内存
    torch.npu.empty_cache()
    
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)