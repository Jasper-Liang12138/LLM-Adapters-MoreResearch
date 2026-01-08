import json
import sys
import os
import re
import copy
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

# ====================== 与 evaluate.py 对齐 ======================
def extract_answer_letter(sentence: str) -> str:
    sentence = sentence.strip()
    preds = re.findall(r'A|B|C|D|E', sentence)
    return preds[0] if preds else ""


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
        "--aqua_path",
        type=str,
        default="dataset/AQuA/test.json",
        help="Path to AQuA test.json",
    )
    parser.add_argument(
        "--dapo_parquet",
        type=str,
        default="dataset/dapo-math-17k.parquet",
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
        "--save_path",
        type=str,
        default="experiment/tf_grpo_aqua_results.json",
        help="Path to save inference results",
    )
    parser.add_argument(
        "--load_8bit",
        default=False,
        action="store_true",
        help="Load model in 8-bit precision to save GPU memory",
    )
    parser.add_argument(
    "--experience_bank_path",
    type=str,
    default=None,
    help="Path to prebuilt experience_bank.json",
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

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

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
        max_experiences=args.exp_size
    )

    # ========== Load or Build Experience Bank ==========
    if args.experience_bank_path and os.path.exists(args.experience_bank_path):
        print(f"[Step 1] Load Experience Bank from {args.experience_bank_path}")
        with open(args.experience_bank_path, "r") as f:
            experience_bank_data = json.load(f)
        tf_grpo.load_experience_bank(experience_bank_data)  # 假设 TF_GRPO 有 load_experience_bank 方法
    else:
        print("[Step 1] Build Experience Bank from DAPO")
        tf_grpo.build_experience_from_dapo_epochs(
            parquet_path=args.dapo_parquet,
            sample_size=args.exp_size,
        )

    # ========== Load AQuA ==========
    print("[Step 2] Load AQuA dataset")
    with open(args.aqua_path, "r") as f:
        dataset = json.load(f)

    total = len(dataset)
    correct = 0
    results = []

    print("[Step 3] TF-GRPO inference on AQuA (batch mode)")

    batch_size = 2  # 每批处理 4 条，可根据显存调
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = dataset[start_idx:end_idx]

        batch_prompts = []
        for data in batch:
            question = data.get("instruction", "")
            input_text = data.get("input", "")
            full_question = question + "\n" + input_text if input_text.strip() else question
            batch_prompts.append(tf_grpo.build_prompt(full_question))

        # ===== TF-GRPO 批量生成 =====
        batch_outputs = tf_grpo.batch_group_generate(batch_prompts)

        # ===== 逐条处理结果 =====
        for i, data in enumerate(batch):
            outputs = batch_outputs[i]
            pred_text = outputs[0]
            pred_letter = extract_answer_letter(pred_text)
            label = data.get("answer", "")

            flag = (pred_letter == label)
            if flag:
                correct += 1

            record = copy.deepcopy(data)
            record["output_pred"] = pred_text
            record["pred"] = pred_letter
            record["flag"] = flag
            results.append(record)

            print("\n---------------")
            print(pred_text)
            print("prediction:", pred_letter)
            print("label:", label)
            print("---------------")
            print(
                f"test:{start_idx + i + 1}/{total} | "
                f"accuracy {correct} {correct / (start_idx + i + 1):.4f}"
            )

        # ===== 每批写入一次结果文件 =====
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=4)

        # 清理显存
        torch.cuda.empty_cache()

    print("\nTest finished")
    print(f"Final AQuA Accuracy: {correct / total:.4f}")
    print(f"Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()





