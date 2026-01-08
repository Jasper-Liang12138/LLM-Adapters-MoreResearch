import sys
import os
import re
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from tf_grpo_dapo import TF_GRPO

# 假设 inference_math.py 在 LLM_Adapters/RLHF/ 下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser("TF-GRPO Inference on AQuA")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base LLM, e.g. yahma/llama-7b-hf",
    )
    parser.add_argument(
        "--dapo_parquet",
        type=str,
        required=True,
        help="DAPO-Math-17K parquet (used only to build experience bank)",
    )
    parser.add_argument(
        "--exp_size",
        type=int,
        default=100,
        help="Number of DAPO samples to build experience bank",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Group rollout size for TF-GRPO",
    )
    parser.add_argument(
        "--load_8bit",
        default=False,
        action="store_true",
        help="Load model in 8-bit precision to save GPU memory",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per step",
    )

    return parser.parse_args()

def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
       
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
       
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model

# ====================== main ======================
def main():
    torch.cuda.empty_cache()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ========== Load model ==========
    print("[Load] Base model")
    tokenizer, model = load_model(args)

    # ========== Init TF-GRPO ==========
    tf_grpo = TF_GRPO(
        model=model,
        tokenizer=tokenizer,
        group_size=args.group_size,
        max_experiences=args.exp_size,
        max_new_tokens=args.max_new_tokens
    )

    # ========== Build Experience Bank ==========
    print("Build Experience Bank from DAPO")
    tf_grpo.build_experience_from_dapo_epochs(
        parquet_path=args.dapo_parquet,
        sample_size=args.exp_size,
    )

    print("\nBuild finished")



if __name__ == "__main__":
    main()





