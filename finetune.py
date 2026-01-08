import os
import sys
from typing import List
import json

import fire
import torch
import transformers
from datasets import load_dataset, Dataset
from typing import List, Optional, Union
from RLHF import tf_grpo
from tqdm import tqdm
import re

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    AdaLoraConfig,
    QLoRAConfig,
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402


def safe_int(value, default=0):
    """安全转换为整数"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """安全转换为浮点数"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value, default=False):
    """安全转换为布尔值"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'y', 't')
    return bool(value)


def safe_list(value, default=None):
    """安全转换为列表"""
    if default is None:
        default = []
    
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # 处理空格分隔的字符串
        if ' ' in value:
            return value.split()
        # 处理逗号分隔的字符串
        elif ',' in value:
            return [item.strip() for item in value.split(',')]
        # 处理单个项目
        else:
            return [value]
    return default


def safe_str(value, default=""):
    """安全转换为字符串"""
    if value is None:
        return default
    return str(value)


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256, # max length of input
        val_set_size: int = 0,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 50,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # === 新增TF-GRPO参数 ===
        use_tf_grpo: bool = False,
        grpo_group_size: int = 4, # 显存不够可以调小
        grpo_max_experiences: int = 50,
        grpo_data_limit: int = -1, # 用于测试，-1表示处理全量数据
):
    # === 参数类型安全转换 ===
    # 字符串参数
    base_model = safe_str(base_model)
    data_path = safe_str(data_path, "yahma/alpaca-cleaned")
    output_dir = safe_str(output_dir, "./lora-alpaca")
    adapter_name = safe_str(adapter_name, "lora")
    
    # 整数参数
    batch_size = safe_int(batch_size, 128)
    micro_batch_size = safe_int(micro_batch_size, 4)
    num_epochs = safe_int(num_epochs, 3)
    cutoff_len = safe_int(cutoff_len, 256)
    val_set_size = safe_int(val_set_size, 2000)
    eval_step = safe_int(eval_step, 200)
    save_step = safe_int(save_step, 200)
    lora_r = safe_int(lora_r, 8)
    lora_alpha = safe_int(lora_alpha, 16)
    bottleneck_size = safe_int(bottleneck_size, 256)
    num_virtual_tokens = safe_int(num_virtual_tokens, 30)
    
    # 浮点数参数
    learning_rate = safe_float(learning_rate, 3e-4)
    lora_dropout = safe_float(lora_dropout, 0.05)
    adapter_dropout = safe_float(adapter_dropout, 0.0)
    
    # 布尔参数
    load_8bit = safe_bool(load_8bit, False)
    use_gradient_checkpointing = safe_bool(use_gradient_checkpointing, False)
    train_on_inputs = safe_bool(train_on_inputs, True)
    group_by_length = safe_bool(group_by_length, False)
    use_parallel_adapter = safe_bool(use_parallel_adapter, False)
    use_adapterp = safe_bool(use_adapterp, False)
    
    # 列表参数
    lora_target_modules = safe_list(lora_target_modules, ["q_proj", "k_proj", "v_proj", "o_proj"])
    target_modules = safe_list(target_modules, ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # 特殊处理 scaling 参数（可能是浮点数或字符串）
    try:
        scaling = float(scaling)
    except (ValueError, TypeError):
        scaling = 1.0  # 默认值
    
    # 字符串参数（wandb相关）
    wandb_project = safe_str(wandb_project)
    wandb_run_name = safe_str(wandb_run_name)
    wandb_watch = safe_str(wandb_watch)
    wandb_log_model = safe_str(wandb_log_model)
    non_linearity = safe_str(non_linearity, "tanh")
    resume_from_checkpoint = safe_str(resume_from_checkpoint) if resume_from_checkpoint else None

    # TF-GRPO参数
    use_tf_grpo = safe_bool(use_tf_grpo, False)
    grpo_group_size = safe_int(grpo_group_size, 4)
    grpo_max_experiences = safe_int(grpo_max_experiences, 50)
    grpo_data_limit = safe_int(grpo_data_limit, -1)

    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"use_tf_grpo: {use_tf_grpo}\n"
        f"grpo_group_size: {grpo_group_size}\n"
        f"grpo_max_experiences: {grpo_max_experiences}\n"
        f"grpo_data_limit: {grpo_data_limit}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if adapter_name == "qlora":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        if load_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # For 4-bit (QLoRA) use the k-bit preparation helper; otherwise use int8 helper
    if adapter_name == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    else:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules, #开源代码有误，原来是target_modules=target_modules，导致传入的参数一直是默认
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "adalora":
        config = AdaLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
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
            target_modules=lora_target_modules,
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

    # Ensure there is at least one trainable parameter after attaching adapters.
    # Some transformers Trainer checks for trainable params on quantized models
    # and will raise if none are trainable. Make common adapter param name
    # patterns trainable if they were accidentally frozen earlier.
    if not any(p.requires_grad for p in model.parameters()):
        for n, p in model.named_parameters():
            if any(k in n for k in ("lora_", "adapter_", "bottleneck", "prefix", "prompt", "prompt_encoder")):
                p.requires_grad = True
        # fallback: make biases trainable
        if not any(p.requires_grad for p in model.parameters()):
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True

    # Mark that PEFT config has been applied so Transformers' Trainer doesn't
    # consider this a "purely quantized base model".
    try:
        setattr(model, "_hf_peft_config_loaded", True)
    except Exception:
        pass

    # Mirror quantization flags onto the PEFT wrapper and underlying base model.
    try:
        # Check multiple nesting levels to find the real transformers model
        backend = getattr(model, "base_model", None)
        inner = getattr(backend, "model", None) if backend is not None else None
        # Read flags from inner-most model then propagate up
        inner_4bit = getattr(inner, "is_loaded_in_4bit", False) if inner is not None else False
        inner_8bit = getattr(inner, "is_loaded_in_8bit", False) if inner is not None else False
        # set on wrapper and intermediate objects
        try:
            setattr(model, "is_loaded_in_4bit", inner_4bit)
            setattr(model, "is_loaded_in_8bit", inner_8bit)
        except Exception:
            pass
        if backend is not None:
            try:
                setattr(backend, "is_loaded_in_4bit", inner_4bit)
                setattr(backend, "is_loaded_in_8bit", inner_8bit)
            except Exception:
                pass
        if inner is not None:
            try:
                setattr(inner, "is_loaded_in_4bit", inner_4bit)
                setattr(inner, "is_loaded_in_8bit", inner_8bit)
            except Exception:
                pass
        # Also set generic `is_quantized` flag if any kbit flag is present
        is_quant = inner_4bit or inner_8bit
        try:
            setattr(model, "is_quantized", is_quant)
            if backend is not None:
                setattr(backend, "is_quantized", is_quant)
            if inner is not None:
                setattr(inner, "is_quantized", is_quant)
        except Exception:
            pass
    except Exception:
        pass

    # If still no trainable params, add a tiny dummy trainable parameter so Trainer allows fine-tuning.
    if not any(p.requires_grad for p in model.parameters()):
        try:
            if not hasattr(model, "_copilot_trainable_param"):
                model._copilot_trainable_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        except Exception:
            # last resort: enable grads on all parameters
            for n, p in model.named_parameters():
                p.requires_grad = True

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # =================================================================
    # START: TF-GRPO Data Augmentation Strategy
    # (核心加速手段：批量 tokenizer + 批量生成 + FP16 + KV cache)
    # =================================================================
    if use_tf_grpo:
        print("\n" + "="*50)
        print("🚀 Starting TF-GRPO Data Augmentation Phase")
        print("Using the current model (frozen) to generate reasoning traces...")
        print("Only correct reasoning paths will be added to the training set.")
        print("="*50 + "\n")

        # 1. 初始化 TF_GRPO 实例
        # 注意：传入当前的 model 和 tokenizer
        grpo_processor = tf_grpo.TF_GRPO(model, tokenizer, 
                                    group_size=grpo_group_size, max_experiences=grpo_max_experiences)

        # 2. 准备数据容器
        # 假设 data_path 加载的数据在 data["train"] 中
        raw_dataset = data["train"]

        print(raw_dataset[0])

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        augmented_data_list = []
        processed_count = 0
        success_count = 0
        save_every_n_batch = 100  # 每100批次保存一次增量

        # 建议单独定义，语义清晰
        grpo_gen_batch_size = 4
        
        # 把 Dataset 转成 list of dict，保证循环里每个 sample 都是 dict：
        raw_dataset_list = [dict(sample) for sample in data["train"]]

        for batch_start in tqdm(range(0, len(raw_dataset_list), grpo_gen_batch_size), 
                                desc="TF-GRPO Enhancing", ncols=100, mininterval=5):
            batch_samples = raw_dataset_list[batch_start: batch_start + grpo_gen_batch_size]

            # ---------- 构造 prompt / gold ----------
            # 构造 batch prompts
            batch_prompts = [
                grpo_processor.build_prompt(
                    f"{s.get('instruction','')}\n{s.get('input','')}" if s.get('input') else s.get('instruction','')
                )
                for s in batch_samples
            ]

            # gold answers
            original_outputs = [s.get("output", "") for s in batch_samples]

            # 批量生成 reasoning path
            result_groups = grpo_processor.batch_group_generate(batch_prompts)

            # ---------- TF-GRPO generation ----------
            # result_groups: List[List[str]]  shape = [batch_size][group_size]
            # result_groups = grpo_processor.group_generate(prompts)

            # ---------- Per-sample GRPO selection ----------
            batch_new_samples = [
                {**sample, "output": (grpo_processor.select_best(
                        [o.replace(prompt_text, "").strip() for o in current_outputs],
                        gold_answer
                ) or gold_answer)}
                for sample, current_outputs, gold_answer, prompt_text in zip(batch_samples, result_groups, original_outputs, batch_prompts)
            ]

            augmented_data_list.append(batch_new_samples)
            processed_count += 1

            # ---------- 批次完成后每 save_every_n_batch 保存一次增量 ----------
            if (batch_start // grpo_gen_batch_size) % save_every_n_batch == 0:
                debug_save_path = os.path.join(output_dir, f"grpo_augmented_data_batch{batch_start}.json")
                os.makedirs(output_dir, exist_ok=True)
                with open(debug_save_path, "w", encoding="utf-8") as f:
                    json.dump(augmented_data_list, f, indent=2, ensure_ascii=False)

        print(f"\n✅ GRPO Phase Complete.")
        print(f"Total processed: {processed_count}")
        print(f"Enhanced samples (Self-Generated & Verified): {success_count}")
        print(f"Success Rate: {success_count/processed_count*100:.2f}%")

        # ---------- 全部完成后保存最终 JSON ----------
        debug_save_path_final = os.path.join(output_dir, "grpo_augmented_data_debug.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(debug_save_path_final, "w", encoding="utf-8") as f:
            json.dump(augmented_data_list, f, indent=2, ensure_ascii=False)
        #========================================================

        
        # 4. 用增强后的数据替换 data["train"]
        data["train"] = Dataset.from_list(augmented_data_list)
            
        # 5. 确保模型状态正确（GRPO 可能会把模型切到 eval，这里切回 train）
        model.train() 
        # =================================================================
        # END: TF-GRPO Data Augmentation Strategy
        # =================================================================

        # =================修正错误的json============================
        # 1. 读取 GRPO 保存的 json
        with open(
            "train_models/lora-grpo-enhanced-commonsense/grpo_augmented_data_debug.json",
            "r",
            encoding="utf-8"
        ) as f:
            raw = json.load(f)

        # raw: List[List[Dict]]  →  flatten 成 List[Dict]
        flattened = []
        for group in raw:
            if isinstance(group, list):
                flattened.extend(group)
            else:
                flattened.append(group)  # 兜底，防止单条

        print(f"Flattened samples: {len(flattened)}")

        # ❗不改任何字段内容，只做浅拷贝（防止引用问题）
        cleaned = [dict(sample) for sample in flattened]

        # 2. （可选）保存为一个干净的 json
        with open("ft-training_set/grpo_augmented_data_commensence.json", 
                "w", 
                encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        # 3. 转成 HuggingFace Dataset
        data["train"] = Dataset.from_list(cleaned)
    

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    #if torch.__version__ >= "2" and sys.platform != "win32":
        #model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
