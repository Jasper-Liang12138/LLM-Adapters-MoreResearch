"""
计算数据集中每个句子的intrinsic dimension
基于LoRA模型的token embeddings，使用PCA方法计算intrinsic dimension
"""
import os
import sys
import json
import argparse

from typing import List, Optional, Union
import fire
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaTokenizer,
    GenerationConfig
)

# 设备设置
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def generate_prompt(instruction, input=None):
    """生成prompt格式"""
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
"""


def load_model(base_model, lora_weights, model_name, load_8bit=False):
    """加载LoRA模型和tokenizer"""
    if "LLaMA" in model_name or "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # unk
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        if not load_8bit:
            model.half()
    
    model.eval()
    return tokenizer, model


def get_token_embeddings(model, tokenizer, text):
    """
    获取句子的token embeddings
    
    Args:
        model: LoRA模型
        tokenizer: tokenizer
        text: 输入文本
    
    Returns:
        embeddings: token embeddings, shape (seq_len, hidden_size)
    """
    # 编码文本
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device) # (batch_size, seq_len)， 每个token的id，to(device)是将tensor移动到指定设备
    attention_mask = inputs["attention_mask"].to(device) # (batch_size, seq_len)， 每个token是否可见，to(device)是将tensor移动到指定设备
    
    with torch.no_grad(): # 禁用梯度计算，节省计算资源
        # 获取word embeddings
        # 对于LoRA模型，从base_model获取embedding层
        try:
            # 尝试从PeftModel获取
            if hasattr(model, 'base_model'):
                base_model = model.base_model
                # 尝试不同的embedding层名称
                if hasattr(base_model, 'model'):
                    model_module = base_model.model
                    if hasattr(model_module, 'embed_tokens'):  # LLaMA
                        embeddings = model_module.embed_tokens(input_ids)
                    elif hasattr(model_module, 'wte'):  # GPT系列
                        embeddings = model_module.wte(input_ids)
                    elif hasattr(model_module, 'word_embeddings'):  # BERT系列
                        embeddings = model_module.word_embeddings(input_ids)
                    else:
                        # 使用通用的get_input_embeddings方法
                        embeddings = model_module.get_input_embeddings()(input_ids) 
                else:
                    # 直接从base_model获取
                    embeddings = base_model.get_input_embeddings()(input_ids)
            else:
                # 如果不是PeftModel，直接获取
                embeddings = model.get_input_embeddings()(input_ids)
        except Exception as e:
            # 如果上述方法都失败，尝试直接访问embedding属性
            print(f"警告: 获取embeddings时出错 {e}，尝试备用方法...")
            if hasattr(model, 'get_input_embeddings'):
                embeddings = model.get_input_embeddings()(input_ids)
            else:
                raise ValueError(f"无法获取模型的embedding层: {e}")
    
    # 移除padding位置的embeddings
    seq_len = attention_mask.sum(dim=1).item() # 计算每个句子的实际长度，attention_mask中为1的token才是有效的，.item()将tensor转换为标量
    embeddings = embeddings[0, :seq_len, :]  # (seq_len, hidden_size)， [0, :seq_len, :]表示只保留前seq_len个token的embeddings
    
    return embeddings.cpu().numpy()


def compute_intrinsic_dimension(embeddings, variance_threshold=0.95):
    """
    计算intrinsic dimension
    
    Args:
        embeddings: token embeddings, shape (n_tokens, hidden_size)
        variance_threshold: 需要保留的方差比例，默认0.95
    
    Returns:
        intrinsic_dim: intrinsic dimension
    """
    if embeddings.shape[0] < 2:
        # 如果token数量太少，无法计算PCA
        return 1
    
    # 标准化数据（按特征标准化）
    embeddings_centered = embeddings - embeddings.mean(axis=0, keepdims=True) 
    
    # 使用PCA计算主成分
    # 如果token数量少于特征数，使用最小维度
    n_components = min(embeddings.shape[0] - 1, embeddings.shape[1])#
    
    if n_components < 1:
        return 1
    
    pca = PCA(n_components=n_components)# 创建PCA对象，n_components为主成分数量
    pca.fit(embeddings_centered)# 拟合PCA模型，embeddings_centered为标准化后的数据
    
    # 计算累计方差解释比例
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)# 计算累计方差解释比例，pca.explained_variance_ratio_为每个主成分的方差解释比例
    
    # 找到第一个达到阈值的主成分数量
    intrinsic_dim = np.argmax(cumsum_variance >= variance_threshold) + 1  #
    
    # 如果所有主成分的累计方差都小于阈值，返回全部维度数
    if cumsum_variance[-1] < variance_threshold:
        intrinsic_dim = len(cumsum_variance)
    
    return intrinsic_dim


def load_data(dataset_name):
    """加载数据集"""
    file_path = f'dataset/{dataset_name}/test.json'
    if not os.path.exists(file_path):
        # 尝试其他可能的文件名
        alt_paths = [
            f'dataset/{dataset_name}/{dataset_name}.json',
            f'dataset/{dataset_name}/aqua_1.json',  # AQuA的特殊情况
            f'dataset/{dataset_name}/gsm8k.json',  # GSM8K的特殊情况
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                file_path = alt_path
                break
        else:
            raise FileNotFoundError(f"无法找到数据集文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def process_dataset(dataset_name, model, tokenizer, variance_threshold=0.95):
    """
    处理整个数据集，计算每个句子的intrinsic dimension
    
    Args:
        dataset_name: 数据集名称
        model: LoRA模型
        tokenizer: tokenizer
        variance_threshold: PCA方差阈值
    
    Returns:
        results: 包含每个样本intrinsic dimension的结果列表
    """
    print(f"\n加载数据集: {dataset_name}")
    dataset = load_data(dataset_name)
    
    results = []
    intrinsic_dims = []
    
    print(f"处理 {len(dataset)} 个样本...")
    for idx, data in enumerate(tqdm(dataset, desc=f"处理 {dataset_name}")):
        # 获取instruction和input
        instruction = data.get('instruction', '')
        input_text = data.get('input', None)
        
        # 生成prompt
        if input_text:
            prompt = generate_prompt(instruction, input_text)
        else:
            prompt = generate_prompt(instruction)
        
        try:
            # 获取token embeddings
            embeddings = get_token_embeddings(model, tokenizer, prompt)
            
            # 计算intrinsic dimension
            intrinsic_dim = compute_intrinsic_dimension(embeddings, variance_threshold)
            
            intrinsic_dims.append(intrinsic_dim)
            
            result_item = {
                'index': idx,
                'instruction': instruction,
                'input': input_text,
                'intrinsic_dimension': int(intrinsic_dim),
            }
            results.append(result_item)
            
        except Exception as e:
            print(f"\n处理样本 {idx} 时出错: {e}")
            continue
    
    # 计算平均值
    avg_intrinsic_dim = np.mean(intrinsic_dims) if intrinsic_dims else 0
    std_intrinsic_dim = np.std(intrinsic_dims) if intrinsic_dims else 0
    median_intrinsic_dim = np.median(intrinsic_dims) if intrinsic_dims else 0
    
    summary = {
        'dataset': dataset_name,
        'total_samples': len(results),
        'average_intrinsic_dimension': float(avg_intrinsic_dim),
        'std_intrinsic_dimension': float(std_intrinsic_dim),
        'median_intrinsic_dimension': float(median_intrinsic_dim),
        'min_intrinsic_dimension': int(np.min(intrinsic_dims)) if intrinsic_dims else 0,
        'max_intrinsic_dimension': int(np.max(intrinsic_dims)) if intrinsic_dims else 0,
    }
    
    return results, summary


def main():
    pass


def compute(
    base_model: str,
    lora_weights: str,
    dataset: Optional[Union[str, List[str]]] = None,
    model_name: str = "LLaMA-7B",
    load_8bit: bool = False,
    variance_threshold: float = 0.95,
    output_dir: str = "intrinsic_dim_results",
    device_arg: Optional[str] = None,
):
    """
    计算指定数据集和微调后模型的 intrinsic dimension。

    参数说明与示例用法请参见仓库中的 `math_running_commands`。
    """
    # normalize dataset arg
    if dataset is None:
        datasets = ["AQuA"]
    elif isinstance(dataset, str):
        datasets = [dataset]
    else:
        datasets = list(dataset)

    # set device if provided
    global device
    if device_arg:
        device = device_arg
    else:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass

    os.makedirs(output_dir, exist_ok=True)

    print("加载模型...")
    tokenizer, model = load_model(base_model, lora_weights, model_name, load_8bit)
    print(f"模型加载完成，设备: {device}")

    all_summaries = []
    for dataset_name in datasets:
        try:
            results, summary = process_dataset(dataset_name, model, tokenizer, variance_threshold)

            output_file = os.path.join(output_dir, f"{dataset_name}_intrinsic_dim.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

            print(f"\n{dataset_name} 结果:")
            print(f"  平均intrinsic dimension: {summary['average_intrinsic_dimension']:.2f}")
            print(f"  标准差: {summary['std_intrinsic_dimension']:.2f}")
            print(f"  中位数: {summary['median_intrinsic_dimension']:.2f}")
            print(f"  最小值: {summary['min_intrinsic_dimension']}")
            print(f"  最大值: {summary['max_intrinsic_dimension']}")
            print(f"  结果已保存到: {output_file}")

            all_summaries.append(summary)
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            continue

    if all_summaries:
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        overall_avg = np.mean([s["average_intrinsic_dimension"] for s in all_summaries])
        print(f"\n所有数据集的平均intrinsic dimension: {overall_avg:.2f}")
        print(f"汇总结果已保存到: {summary_file}")


if __name__ == "__main__":
    fire.Fire(compute)

