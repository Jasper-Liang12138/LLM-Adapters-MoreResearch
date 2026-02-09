import os
import sys

# 1. 依然保留 NPU 环境变量，防止算子底层转 FP16
os.environ["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu # 自动将 cuda 调用转为 npu
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

def train(
    base_model: str = "Qwen/Qwen2.5-7B", 
    data_path: str = "",
    output_dir: str = "./lora-qwen-ds", 
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list = None,
    resume_from_checkpoint: str = None,
    # 这里的 local_rank 参数对 DeepSpeed 很重要
    local_rank: int = -1,
    **kwargs 
):
    print(f"Loading Model: {base_model}...")

    # 2. 不要在代码里指定 device_map="auto" 或者 .to(device)
    # DeepSpeed 会自动管理模型分配
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        #以此确保模型不会自动乱分卡
        device_map=None 
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 数据处理逻辑 (简略版，和你之前的一样)
    def generate_and_tokenize_prompt(data_point):
        # 请确保你的数据字段 key 是正确的 (instruction/input/output)
        instruction = data_point.get("instruction", "")
        input_text = data_point.get("input", "")
        output_text = data_point.get("output", "")
        
        text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
        tokenized = tokenizer(text, truncation=True, max_length=cutoff_len, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, 
        target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM"
    )
    
    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, config)
    
    # 再次确保参数可训练
    model.print_trainable_parameters()
    model.config.use_cache = False

    # 加载数据
    data = load_dataset("json", data_files=data_path)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    
    # 3. TrainingArguments 配置
    # DeepSpeed 通过 args 传入，transformers会自动检测并初始化 DeepSpeed引擎
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=(batch_size // micro_batch_size) // int(os.environ.get("WORLD_SIZE", 1)),
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,   # 这里的 True 主要是给 Trainer 看的
        fp16=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        # ！！！关键点！！！
        deepspeed="./ds_config.json" 
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
    )
    
    # 启动训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import fire
    fire.Fire(train)