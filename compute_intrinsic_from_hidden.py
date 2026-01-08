import os
import sys
import json
from typing import List, Optional, Union

import fire
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


# device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass


def _load_data(dataset_name: str):
    file_path = f'dataset/{dataset_name}/test.json'
    if not os.path.exists(file_path):
        alt_paths = [
            f'dataset/{dataset_name}/{dataset_name}.json',
            f'dataset/{dataset_name}/aqua_1.json',
            f'dataset/{dataset_name}/gsm8k.json',
        ]
        for p in alt_paths:
            if os.path.exists(p):
                file_path = p
                break
        else:
            raise FileNotFoundError(f"无法找到数据集文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_model(base_model: str, lora_weights: str = None, model_name: str = "LLaMA-7B", load_8bit: bool = False):
    if "llama" in model_name.lower() or "LLaMA" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": device})
        if lora_weights:
            model = PeftModel.from_pretrained(model, lora_weights, device_map={"": device})

    model.eval()
    return tokenizer, model


def _get_last_hidden_states(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Try to get underlying base model for forward to ensure we get hidden states
    base = None
    if hasattr(model, 'get_base_model'):
        try:
            base = model.get_base_model()
        except Exception:
            base = None
    if base is None:
        # fallback to common attributes
        base = getattr(model, 'base_model', model)

    with torch.no_grad():
        outputs = base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        # try to get last_hidden_state, else fallback to hidden_states[-1]
        last_hidden = getattr(outputs, 'last_hidden_state', None)
        if last_hidden is None:
            hidden_states = getattr(outputs, 'hidden_states', None)
            if hidden_states:
                last_hidden = hidden_states[-1]
            else:
                raise RuntimeError('模型未返回 last_hidden_state 或 hidden_states，无法计算 hidden 表示')

    seq_len = attention_mask.sum(dim=1).item()
    hidden = last_hidden[0, :seq_len, :].cpu().numpy()
    return hidden


def compute_intrinsic_dim_from_hidden(hidden: np.ndarray, variance_threshold: float = 0.95) -> int:
    if hidden.shape[0] < 2:
        return 1
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    n_components = min(hidden.shape[0] - 1, hidden.shape[1])
    if n_components < 1:
        return 1
    pca = PCA(n_components=n_components)
    pca.fit(centered)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.argmax(cumsum >= variance_threshold) + 1)
    if cumsum[-1] < variance_threshold:
        idx = len(cumsum)
    return idx


def compute(
    base_model: str,
    lora_weights: Optional[str] = None,
    dataset: Union[str, List[str]] = "AQuA",
    model_name: str = "LLaMA-7B",
    load_8bit: bool = False,
    variance_threshold: float = 0.95,
    output_dir: str = "intrinsic_hidden_results",
    device_arg: Optional[str] = None,
):
    """Compute intrinsic dimension per-sentence from last hidden state and summarize per-dataset."""
    global device
    if device_arg:
        device = device_arg

    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        datasets = list(dataset)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    tokenizer, model = load_model(base_model, lora_weights, model_name, load_8bit)
    print(f"Model loaded on device: {device}")

    all_summaries = []
    for ds in datasets:
        data = _load_data(ds)
        dims = []
        results = []
        for i, item in enumerate(tqdm(data, desc=f"Processing {ds}")):
            instruction = item.get('instruction', '')
            input_text = item.get('input', None)
            if input_text:
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            try:
                hidden = _get_last_hidden_states(model, tokenizer, prompt)
                dim = compute_intrinsic_dim_from_hidden(hidden, variance_threshold)
                dims.append(int(dim))
                results.append({'index': i, 'intrinsic_dimension': int(dim)})
            except Exception as e:
                results.append({'index': i, 'error': str(e)})
                continue

        summary = {
            'dataset': ds,
            'total': len(results),
            'average_intrinsic_dimension': float(np.mean(dims)) if dims else 0.0,
            'std_intrinsic_dimension': float(np.std(dims)) if dims else 0.0,
            'median_intrinsic_dimension': float(np.median(dims)) if dims else 0.0,
            'min_intrinsic_dimension': int(np.min(dims)) if dims else 0,
            'max_intrinsic_dimension': int(np.max(dims)) if dims else 0,
        }

        out_file = os.path.join(output_dir, f"{ds}_hidden_intrinsic.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'results': results}, f, indent=2, ensure_ascii=False)

        print(f"{ds} summary: avg={summary['average_intrinsic_dimension']:.2f} median={summary['median_intrinsic_dimension']:.2f} saved={out_file}")
        all_summaries.append(summary)

    if all_summaries:
        summary_file = os.path.join(output_dir, 'all_summary_hidden.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"All summaries saved to {summary_file}")


if __name__ == '__main__':
    fire.Fire(compute)
