import re
import random
from typing import List
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from transformers import StoppingCriteria, StoppingCriteriaList


class StopSequenceCriteria(StoppingCriteria):
    """自定义停止条件：遇到停止序列时停止生成"""
    def __init__(self, tokenizer, stop_sequences: List[str], prompt_lengths: List[int]):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.prompt_lengths = prompt_lengths
        # 将停止序列转换为 token ids
        self.stop_token_ids = []
        for seq in stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            if tokens:
                self.stop_token_ids.append(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = input_ids.shape[0]
        
        # 检查每个样本是否满足停止条件
        for i in range(batch_size):
            prompt_len = self.prompt_lengths[i] if i < len(self.prompt_lengths) else self.prompt_lengths[0]
            generated = input_ids[i, prompt_len:].tolist()
            
            if len(generated) == 0:
                continue
            
            # 检查末尾是否匹配任何停止序列
            for stop_ids in self.stop_token_ids:
                if len(generated) >= len(stop_ids):
                    # 检查最后几个 token 是否匹配停止序列
                    if generated[-len(stop_ids):] == stop_ids:
                        return True
        
        return False


class TF_GRPO:
    def __init__(self, model, tokenizer, group_size=8, max_experiences=100, max_new_tokens=128):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_experiences = max_experiences
        self.max_new_tokens = max_new_tokens
        self.experience_bank = []
        self.exp_embeddings = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    # ===================== 计算文本 embedding =====================
    def embed_func(self, text: str):
        # tokenization
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # 获取隐藏层表征
        with torch.no_grad():
            # 尝试获取 base_model 以确保能获取 hidden states
            base = None
            if hasattr(self.model, 'get_base_model'):
                try:
                    base = self.model.get_base_model()
                except Exception:
                    base = None
            if base is None:
                # fallback to common attributes
                base = getattr(self.model, 'base_model', self.model)
            
            # 通过 forward pass 获取隐藏层输出
            outputs = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 获取最后一层的隐藏状态
            last_hidden = getattr(outputs, 'last_hidden_state', None)
            if last_hidden is None:
                hidden_states = getattr(outputs, 'hidden_states', None)
                if hidden_states:
                    last_hidden = hidden_states[-1]  # [1, seq_len, hidden_size]
                else:
                    raise RuntimeError('模型未返回 last_hidden_state 或 hidden_states，无法计算隐藏层表征')
            
            # 使用 attention_mask 进行 mean pooling（只对有效 token 取平均）
            if attention_mask is not None:
                # 扩展 attention_mask 维度用于广播: [1, seq_len] -> [1, seq_len, 1]
                mask_expanded = attention_mask.unsqueeze(-1).float()
                # 将 padding 位置的 hidden state 置为 0
                masked_hidden = last_hidden * mask_expanded
                # 对有效 token 取平均
                sum_hidden = masked_hidden.sum(dim=1)  # [1, hidden_size]
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [1, 1]
                sent_emb = sum_hidden / seq_lengths  # [1, hidden_size]
            else:
                # 如果没有 attention_mask，直接对所有 token 取平均
                sent_emb = last_hidden.mean(dim=1)  # [1, hidden_size]
        
        return sent_emb.cpu().numpy()[0]  # 返回 1D np.array

    # ===================== pθ(y | x, E) =====================
    def build_prompt(self, question: str) -> str:
        prompt = (
            "Please solve the math problems and give a final answer at last.\n"
        )
        prompt += (f"Solve the following problem step by step. Use at most {self.max_new_tokens} tokens.\n"
                    "Only output the final answer as a number or fraction in the Answer line.\n"
                    "Output should be:\n"
                    "Step 1: <concise reasoning in words>\n"
                    "Step 2: <concise reasoning in words>\n"
                    "Step 3: <concise reasoning in words>\n"
                    "Answer: <numerical answer here>\n\n"
        
                    f"Problem:\n{question}\n\n"
                    )
        
        return prompt
    # ===================== rollout × G =====================
    def extract_similar_experiences(self, question: str, top_k) -> List[str]:
        # 取 与quiry 最相似的 top_k 条经验
        if self.experience_bank and self.exp_embeddings:
             # 计算问题 embedding
            q_emb = self.embed_func(question).reshape(1, -1)
            # 计算与经验库的相似度, (1, N)点乘(N, D) -> (1, N)
            # 因为 cosine_similarity 返回的是二维数组（即使只有 1 行），[0] 表示取这个二维数组的第一行（也是唯一一行）
            sims = cosine_similarity(q_emb, np.vstack(self.exp_embeddings))[0]
            # 取 top_k
            # argsort 返回的是升序排序的索引，[-top_k:]取倒数第 k 个元素，即为相似度最大的 top_k 个索引
            # [::-1] 表示将这些索引反转，变成降序排列
            top_indices = sims.argsort()[-top_k:][::-1]
            # 相似度阈值
            similarity_threshold = 0.1  # 可以根据需要调整
            top_experiences = [self.experience_bank[i] for i in top_indices if sims[i] >= similarity_threshold]
        else:
            # 经验库为空时 fallback: 取最近的几条
            top_experiences = self.experience_bank[-top_k:]
        return top_experiences
        
    # ===================== generate =====================
    @torch.no_grad()
    def batch_group_generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1500
        ).to(self.model.device)

        # 定义停止序列：遇到这些序列时停止生成
        # 注意：停止序列会在生成过程中匹配，一旦匹配就停止
        stop_sequences = [
            "\n\n\n",  # 多个连续换行（表示内容结束）
        ]
        
        # 计算每个 prompt 的实际长度（排除 padding）
        prompt_lengths = []
        for i in range(len(prompts)):
            # 找到非 padding token 的长度
            input_ids_i = inputs["input_ids"][i]
            if self.tokenizer.pad_token_id is not None:
                # 找到第一个非 padding token 的位置（从右往左）
                non_pad_mask = (input_ids_i != self.tokenizer.pad_token_id)
                if non_pad_mask.any():
                    prompt_len = non_pad_mask.nonzero()[0][-1].item() + 1
                else:
                    prompt_len = len(input_ids_i)
            else:
                prompt_len = len(input_ids_i)
            prompt_lengths.append(prompt_len)
        
        # 创建停止条件
        stop_criteria = StopSequenceCriteria(
            self.tokenizer,
            stop_sequences,
            prompt_lengths
        )
        stopping_criteria = StoppingCriteriaList([stop_criteria])

        # 设置较大的 max_new_tokens 作为安全上限，但主要依靠停止序列
        max_tokens_limit = max(self.max_new_tokens * 4, 512)  # 至少是原来的4倍或512

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_tokens_limit,  # 安全上限，防止无限生成
            num_return_sequences=1,  # 每个输入只生成一条
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,  # 遇到 EOS token 时停止
            stopping_criteria=stopping_criteria,  # 使用自定义停止条件
        )

        # 直接解码输出
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # ===================== GRPO =====================
    def extract_answer(self, text: str) -> str:
        patterns = [r"Answer\s*[:\-]?\s*([^\n$]+)"]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                ans = ans.replace("$", "").replace("\\", "").strip()
                # 匹配数字或分数
                if re.match(r"^[\d\.\-/]+$", ans):
                    return ans
        # fallback: 最后一行纯数字或分数
        for l in reversed(text.splitlines()):
            l_clean = l.replace("$","").replace("\\","").strip()
            if re.match(r"^[\d\.\-/]+$", l_clean):
                return l_clean
        return ""  # 如果没有数字，返回空



    def reasoning_quality(self, reasoning: str) -> float:
        score = 0.0

        # 数学符号
        score += reasoning.count("=") * 0.3
        score += reasoning.count("\\") * 0.2  # LaTeX
        score += reasoning.count("^") * 0.2

        # 逻辑词（偏数学）
        logic_words = [
            "let", "assume", "then", "therefore", "hence",
            "because", "so", "implies", "we have", "consider",
            "case", "thus", "suppose"
        ]
        for w in logic_words:
            score += reasoning.lower().count(w) * 0.5

        # 惩罚垃圾
        if "<p>" in reasoning:
            score -= 10
        if "You are a reasoning assistant" in reasoning:
            score -= 15

        return score

    def reward(self, output: str, gold: str) -> float:
        answer = self.extract_answer(output)
        answer_reward = float(answer == gold)  # EM
        reasoning_reward = self.reasoning_quality(output)
        # 可以调整权重
        alpha, beta = 0.6, 0.4
        return alpha * answer_reward + beta * reasoning_reward

    def select_best(self, outputs: List[str], gold: str) -> str:
        # 先过滤明显无效的
        # valid_outputs = [o for o in outputs if self.is_valid_reasoning(o)]

        #if not valid_outputs:
            #return None  # ❗直接丢弃这个样本
        # 拆分 output
        #reasonings = self.split_output_into_reasonings(outputs)
        
        scores = [self.reward(o, gold) for o in outputs]

        if max(scores) > 0:
            return outputs[scores.index(max(scores))]

        # fallback：不是最长，而是「最像推理的」, 只看推理部分，不考虑EM
        return max(outputs, key=self.reasoning_quality)


    def extract_reasoning_from_output(self, prompt: str, output: str) -> str:
        """
        去掉 prompt 部分，只保留模型生成的 reasoning 内容
        """
        # 如果 output 里包含 prompt, 直接去掉
        reasoning = output.replace(prompt, "")
        # 仅删除开头连续的 [Experience X] 块
        # reasoning = re.sub(r"^(?:\[Experience \d+\].*?\n)+", "", reasoning, flags=re.DOTALL)
        # 3. 清理 HTML 标签
        reasoning = re.sub(r"<[^>]+>", " ", reasoning)
        # 移除 Answer 行
        # reasoning = re.sub(r"(?:Answer|Final Answer)\s*[:\-]?\s*.*", "", reasoning, flags=re.IGNORECASE)
        # 移除连续重复公式
        # reasoning = re.sub(r"(\S+\s*){2,}", r"\1", reasoning)
        # 移除多余空行
        reasoning = re.sub(r"\n+", "\n", reasoning).strip()
        return reasoning


    def update_experience(self, problem: str, reasoning: str, gold: str):
    # 去掉空或无效内容
        reasoning = reasoning.strip()
        if not reasoning:
            return
        # ========== 3. 统一写入格式 ==========
        experience = (
            f"Problem:\n{problem}\n\n"
            f"Experience:\n{reasoning}\n\n"
            f"Gold Answer:\n{gold}"
        )

        self.experience_bank.append(experience)

        # ========== 4. 维护容量 ==========
        if len(self.experience_bank) > self.max_experiences:
            self.experience_bank.pop(0)

    def is_valid_reasoning(self, text: str) -> bool:
        if not text:
            return False

        text = text.strip()
        bad_patterns = [
            "You are a reasoning assistant",
            "The following are useful reasoning experiences",
        ]
         # 只有：包含这两句之一 + 文本很短，才判不合规
        if len(text) < 100:
            for p in bad_patterns:
                if p in text:
                    return False
        return True

    # ===================== 从外部加载经验库 =====================
    def load_experience_bank(self, experience_data: list):
        """
        Load experience bank from external list of reasoning strings.

        Args:
            experience_data (list[str]): List of reasoning examples.
        """
        if not isinstance(experience_data, list):
            raise ValueError("experience_data must be a list of strings.")
        # 仅保留最近 max_experiences 条
        self.experience_bank = experience_data[-self.max_experiences:]

        # 同时计算 embedding
        self.exp_embeddings = []
        print(f"[TF-GRPO] Loading {len(self.experience_bank)} experiences...")
        for exp in tqdm(self.experience_bank, desc="Embedding experiences"):
            emb = self.embed_func(exp)
            self.exp_embeddings.append(emb)
        print("[TF-GRPO] Experience bank loaded.")

    # ===================== Build E from DAPO =====================
    @torch.no_grad()
    def build_experience_from_dapo(self, parquet_path: str, sample_size=100):
        df = pd.read_parquet(parquet_path)
        records = df.to_dict(orient="records")
        samples = random.sample(records, min(sample_size, len(records)))

        for item in tqdm(samples):
            q, a = item["prompt"][0]["content"], item["reward_model"]["ground_truth"]
            prompt = self.build_prompt(q)
            outs = self.batch_group_generate([prompt])[0]
            outs = [o.replace(prompt, "").strip() for o in outs]
            best = self.select_best(outs, a)
            if best is None:
                continue   # 跳过该样本，不写入 bank
            self.update_experience(q, best, a)
        experience_bank = self.experience_bank
        # 保存到文件
        with open("RLHF/experience_bank.json", "w", encoding="utf-8") as f:
            json.dump(experience_bank, f, ensure_ascii=False, indent=2)
    # ===================== Build E from DAPO with Epochs =====================
    @torch.no_grad()
    def build_experience_from_dapo_epochs(
        self,
        parquet_path: str,
        sample_size=100,
        epochs=3,
    ):
        print_samples=100
        df = pd.read_parquet(parquet_path)
        records = df.to_dict(orient="records")

        for ep in range(epochs):
            print(f"\n[TF-GRPO] Epoch {ep+1}/{epochs}")

            samples = random.sample(
                records, min(sample_size, len(records))
            )

            for idx,item in enumerate(tqdm(samples)):
                q, a = item["prompt"][0]["content"], item["reward_model"]["ground_truth"]

                prompt = self.build_prompt(q)
                top_experiences = self.extract_similar_experiences(q, self.group_size)
                outputs = []  # 用来收集每个经验生成的 output
                if not top_experiences:
                    out = self.batch_group_generate([prompt])[0]
                    reasoning = self.extract_reasoning_from_output(prompt, out)
                    if reasoning:
                        outputs.append(reasoning)
                else:
                    # 添加经验到 prompt
                    for idx,exp in enumerate(top_experiences, 1):
                        prompt += "Use the following reasoning experiences internally to help your solution, " \
                                    "but do NOT copy them verbatim into your answer.\n"
                        prompt += f"[Experience {idx}]\n{exp}\n\n"
                        out = self.batch_group_generate([prompt])[0]
                        reasoning = self.extract_reasoning_from_output(prompt, out)
                        if reasoning:    
                            outputs.append(reasoning)

                best = self.select_best(outputs, a)
                if best is None:
                    continue   # 跳过该样本，不写入 bank

                self.update_experience(q, best, a)

                # 打印验证
                if idx < print_samples:
                    print(f"\nSample {idx+1}")
                    print("Question:", q)
                    #print("outputs:", outs)
                    print("Selected reasoning:", best)
                    print("extracted answer:", self.extract_answer(best))
                    print("Gold answer:", a)

            experience_bank = self.experience_bank
            # 阶段性保存到文件
            with open(f"RLHF/experience_bank{ep}.json", "w", encoding="utf-8") as f:
                json.dump(experience_bank, f, ensure_ascii=False, indent=2)

        experience_bank = self.experience_bank
        # 保存到文件
        with open("RLHF/experience_bank.json", "w", encoding="utf-8") as f:
            json.dump(experience_bank, f, ensure_ascii=False, indent=2)

        
