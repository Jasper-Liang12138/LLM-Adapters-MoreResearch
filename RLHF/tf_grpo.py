import re
import random
from typing import List, Dict, Optional
import torch

class TF_GRPO:
    """
    True Training-Free Group Relative Policy Optimization

    Key properties:
    - Model parameters are completely frozen
    - Optimization happens in context / experience space
    - Dataset labels are used only for relative ranking
    """

    def __init__(
        self,
        model,
        tokenizer,
        group_size: int = 8,
        max_experiences: int = 50, 
        device: str = "cuda",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.group_size = group_size 
        self.max_experiences = max_experiences # maximum number of experiences to keep in the experience bank
        self.device = model.device

        self.model.to(self.device)

        # Experience Bank: stores good reasoning traces
        self.experience_bank: List[str] = []

        # Hard-freeze parameters (conceptually important)
        # for p in self.model.parameters():
        #    p.requires_grad_(False)

# ============================构造 GRPO Prompt================================
    def build_prompt(self, question: str) -> str:
        """
         Build prompt with experience-based policy prior. P = f(x, E)
        """
        # Add question to experience bank
        prompt = "You are a helpful assistant that solves math word problems.\n\n"
        
        # Add previous experience traces, E
        if len(self.experience_bank) > 0:
            prompt += "Here are examples of effective reasoning:\n\n"
            for i, exp in enumerate(self.experience_bank[-5:], 1): #[-5:] 取最近的5个experience traces
                prompt += f"Example {i}:\n{exp}\n\n"

        # Add current problem, X
        prompt += f"Now solve the following problem step by step:\n{question}\nAnswer:"
        return prompt
    
# ================Group Rollouts（同一prompt，多次生成）=================    
    '''
    @torch.no_grad()
    def group_generate(self, prompts: List[str]) -> List[List[str]]:
        was_training = self.model.training
        self.model.eval()

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        result_groups = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self.model.device)

            group_outputs = []

            for _ in range(self.group_size):
                out = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=128,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    num_return_sequences=self.group_size,  # 一次生成多条, 比循环生成快
                )

                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                group_outputs.append(text)

            result_groups.append(group_outputs)

        if was_training:
            self.model.train()

        return result_groups
    '''
    @torch.no_grad()
    def batch_group_generate(self, batch_prompts: list):
        """批量生成每个 prompt 的 group_size 条 reasoning path"""
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.model.device)

        # 使用 fp16 速度更快
        model = self.model.half()

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            num_return_sequences=self.group_size
        )

        # reshape 输出 shape = [batch_size, group_size]
        batch_size = len(batch_prompts)
        out = outputs.view(batch_size, self.group_size, -1)

        result_groups = [
            [self.tokenizer.decode(o, skip_special_tokens=True) for o in grp] for grp in out
        ]
        return result_groups


    
# ===================Reward（只做relative ranking）================== 
    def extract_answer(self, text: str):
        """
        Extract a model's final answer in a task-agnostic manner.
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()

        # 1. Look for common "final answer" cues (task-agnostic)
        patterns = [
            r"(?:final\s*answer|answer|conclusion)\s*[:\-]\s*(.+)",
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # 2. Fallback: last non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            return lines[-1]

        # 3. Ultimate fallback: entire output
        return text


    def reward(self, output: str, reference):
        """
        Generic reward function for relative ranking.
        """
        pred = self.extract_answer(output)

        # Case 1: exact match (math, short answer)
        if isinstance(reference, str):
            return float(pred.strip() == reference.strip())

        # Case 2: multiple choice (A/B/C/D)
        if isinstance(reference, dict) and "label" in reference:
            return float(reference["label"] in pred)

        # Case 3: rule-based verifier
        if callable(reference):
            return float(reference(pred))

        # Case 4: no reference (unsupervised heuristic)
        return len(output) / 1000.0  # length-based fallback

    
# ===================Group-Relative Selection（GRPO 核心）====================  
    def select_best(self, outputs: List[str], gold_answer: str) -> str:
        rewards = [self.reward(o, gold_answer) for o in outputs]

        # Prefer correct solutions
        if max(rewards) > 0:
            idx = rewards.index(max(rewards))
            return outputs[idx]

        # If all wrong, fallback to longest reasoning (heuristic)
        return max(outputs, key=len)
    
# ====================Experience Bank Update（Policy Improvement）====================    
    def update_experience(self, reasoning: str):
        self.experience_bank.append(reasoning)

        if len(self.experience_bank) > self.max_experiences:
            self.experience_bank.pop(0)

# ===================【新增接口】供 finetune.py 调用============================
    def process_single_item(self, question: str, gold_answer: str) -> Optional[str]:
        """
        输入问题和标准答案，返回通过 GRPO 优化后的推理文本。
        如果没生成对的，返回 None。
        """
        prompt = self.build_prompt(question)
        outputs = self.group_generate(prompt)
        
        # 清洗 output，去掉 prompt 部分
        cleaned_outputs = []
        for o in outputs:
            if prompt in o:
                cleaned_outputs.append(o.replace(prompt, "").strip())
            else:
                cleaned_outputs.append(o) # Fallback

        best_reasoning = self.select_best(cleaned_outputs, gold_answer)
        
        if best_reasoning:
            self.update_experience(best_reasoning)
            return best_reasoning
        return None






