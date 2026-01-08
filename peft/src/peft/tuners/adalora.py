# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class AdaLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        r (`int`): AdaLoRA rank (initial rank, will be adapted during training)
        target_modules (`Union[List[str],str]`): The names of the modules to apply AdaLoRA to.
        lora_alpha (`float`): The alpha parameter for AdaLoRA scaling.
        lora_dropout (`float`): The dropout probability for AdaLoRA layers.
        init_r (`int`): Initial rank for AdaLoRA (default: 12)
        target_r (`int`): Target rank for AdaLoRA (default: 8)
        beta1 (`float`): Beta1 for importance score (default: 0.85)
        beta2 (`float`): Beta2 for importance score (default: 0.85)
        tinit (`int`): Initial training steps before rank adaptation (default: 200)
        tfinal (`int`): Final training steps for rank adaptation (default: 1000)
        deltaT (`int`): Interval for rank adaptation (default: 10)
        lora_dropout (`float`): The dropout probability for AdaLoRA layers.
        merge_weights (`bool`):
            Whether to merge the weights of the AdaLoRA layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        bias (`str`): Bias type for AdaLoRA. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from AdaLoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "AdaLoRA target rank"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with AdaLoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "AdaLoRA alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "AdaLoRA dropout"})
    # AdaLoRA rank adaptation parameters
    init_r: int = field(default=12, metadata={"help": "Initial rank for AdaLoRA"}) # Initial rank for AdaLoRA (default: 12)
    target_r: int = field(default=8, metadata={"help": "Target rank for AdaLoRA"}) # Target rank for AdaLoRA (default: 8)
    beta1: float = field(default=0.85, metadata={"help": "Beta1 for importance score"}) # Beta1 for importance score (default: 0.85)
    beta2: float = field(default=0.85, metadata={"help": "Beta2 for importance score"}) # Beta2 for importance score (default: 0.85)
    tinit: int = field(default=200, metadata={"help": "Initial training steps before rank adaptation"}) # Initial training steps before rank adaptation (default: 200)
    tfinal: int = field(default=1000, metadata={"help": "Final training steps for rank adaptation"}) # Final training steps for rank adaptation (default: 1000)
    deltaT: int = field(default=10, metadata={"help": "Interval for rank adaptation"}) # Interval for rank adaptation (default: 10)
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the AdaLoRA model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for AdaLoRA. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from AdaLoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADALORA


class AdaLoraModel(torch.nn.Module):
    """
    Creates Adaptive Low Rank Adapter (AdaLoRA) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLoRA model.

    Returns:
        `torch.nn.Module`: The AdaLoRA model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, AdaLoraConfig >>> from peft import AdaLoraModel, AdaLoraConfig >>>
        config = AdaLoraConfig(
            peft_type="ADALORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> adalora_model = AdaLoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLoRA model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_adalora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use AdaLoRA with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "init_r": self.peft_config.init_r,
            "target_r": self.peft_config.target_r,
            "beta1": self.peft_config.beta1,
            "beta2": self.peft_config.beta2,
            "tinit": self.peft_config.tinit,
            "tfinal": self.peft_config.tfinal,
            "deltaT": self.peft_config.deltaT,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear):
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "adalora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, AdaLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on AdaLoRA paper: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
# and modified to work with PyTorch FSDP


def mark_only_adalora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and "adalora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, AdaLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class AdaLoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        init_r: int = 12,
        target_r: int = 8,
        beta1: float = 0.85,
        beta2: float = 0.85,
        tinit: int = 200,
        tfinal: int = 1000,
        deltaT: int = 10,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.init_r = init_r
        self.target_r = target_r
        self.beta1 = beta1
        self.beta2 = beta2
        self.tinit = tinit
        self.tfinal = tfinal
        self.deltaT = deltaT
        self.current_step = 0
        
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, AdaLoraLayer):
    # AdaLoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        init_r: int = 12,
        target_r: int = 8,
        beta1: float = 0.85,
        beta2: float = 0.85,
        tinit: int = 200,
        tfinal: int = 1000,
        deltaT: int = 10,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        AdaLoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            init_r=init_r,
            target_r=target_r,
            beta1=beta1,
            beta2=beta2,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters using SVD decomposition
        if r > 0:
            # Use SVD: W ≈ PΛQ^T
            # P: (out_features, init_r), Q: (in_features, init_r), Lambda: (init_r,)
            self.adalora_P = nn.Parameter(torch.zeros(out_features, init_r))
            self.adalora_Q = nn.Parameter(torch.zeros(in_features, init_r))
            self.adalora_E = nn.Parameter(torch.zeros(init_r))  # Singular values (Lambda)
            
            # Importance scores for rank adaptation
            self.importance = torch.zeros(init_r, device=self.weight.device)
            
            self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "adalora_P"):
            # Initialize P and Q with orthogonal matrices
            nn.init.orthogonal_(self.adalora_P)
            nn.init.orthogonal_(self.adalora_Q)
            # Initialize singular values with small values
            nn.init.normal_(self.adalora_E, mean=0.0, std=0.02)
            self.importance.zero_()

    def update_importance(self):
        """Update importance scores based on gradient information"""
        if hasattr(self, "adalora_P") and hasattr(self, "adalora_Q") and hasattr(self, "adalora_E"):
            # Compute importance as |E| * (||P||_F + ||Q||_F)
            P_norm = torch.norm(self.adalora_P, dim=0)
            Q_norm = torch.norm(self.adalora_Q, dim=0)
            self.importance = torch.abs(self.adalora_E) * (P_norm + Q_norm)

    def prune_rank(self):
        """Prune ranks based on importance scores"""
        if not hasattr(self, "adalora_P"):
            return
        
        current_r = self.adalora_P.shape[1]
        if current_r <= self.target_r:
            return
        
        # Get indices of least important ranks
        _, indices = torch.sort(self.importance, descending=True)
        keep_indices = indices[:self.target_r]
        
        # Prune P, Q, and E
        self.adalora_P = nn.Parameter(self.adalora_P[:, keep_indices])
        self.adalora_Q = nn.Parameter(self.adalora_Q[:, keep_indices])
        self.adalora_E = nn.Parameter(self.adalora_E[keep_indices])
        self.importance = self.importance[keep_indices]

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if hasattr(self, "adalora_P"):
            self.adalora_P.requires_grad = mode
            self.adalora_Q.requires_grad = mode
            self.adalora_E.requires_grad = mode
        
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and hasattr(self, "adalora_P"):
                # W = P * diag(E) * Q^T
                delta_w = self.adalora_P @ torch.diag(self.adalora_E) @ self.adalora_Q.T
                self.weight.data += transpose(delta_w, self.fan_in_fan_out) * self.scaling
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and hasattr(self, "adalora_P"):
                delta_w = self.adalora_P @ torch.diag(self.adalora_E) @ self.adalora_Q.T
                self.weight.data -= transpose(delta_w, self.fan_in_fan_out) * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if hasattr(self, "adalora_P"):
            self.adalora_P.requires_grad = False
            self.adalora_Q.requires_grad = False
            self.adalora_E.requires_grad = False

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            if self.r > 0 and self.merged and hasattr(self, "adalora_P"):
                delta_w = self.adalora_P @ torch.diag(self.adalora_E) @ self.adalora_Q.T
                self.weight.data -= transpose(delta_w.to(previous_dtype), self.fan_in_fan_out) * self.scaling
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged and hasattr(self, "adalora_P"):
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            # AdaLoRA forward: x -> Q -> diag(E) -> P -> output
            x_dropout = self.lora_dropout(x.to(self.adalora_Q.dtype))
            after_Q = F.linear(x_dropout, self.adalora_Q.T)  # (batch, init_r)
            after_E = after_Q * self.adalora_E  # Element-wise multiplication with singular values
            after_P = F.linear(after_E, self.adalora_P)  # (batch, out_features)
            result += after_P.to(previous_dtype) * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, AdaLoraLayer):
        # AdaLoRA implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_r: int = 12,
            target_r: int = 8,
            beta1: float = 0.85,
            beta2: float = 0.85,
            tinit: int = 200,
            tfinal: int = 1000,
            deltaT: int = 10,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            AdaLoraLayer.__init__(
                self,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=False,
                init_r=init_r,
                target_r=target_r,
                beta1=beta1,
                beta2=beta2,
                tinit=tinit,
                tfinal=tfinal,
                deltaT=deltaT,
            )
            # Actual trainable parameters using SVD decomposition
            if r > 0:
                self.adalora_P = nn.Parameter(torch.zeros(out_features, init_r))
                self.adalora_Q = nn.Parameter(torch.zeros(in_features, init_r))
                self.adalora_E = nn.Parameter(torch.zeros(init_r))
                self.importance = torch.zeros(init_r, device=self.weight.device)
                self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
                self.weight.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "adalora_P"):
                nn.init.orthogonal_(self.adalora_P)
                nn.init.orthogonal_(self.adalora_Q)
                nn.init.normal_(self.adalora_E, mean=0.0, std=0.02)
                self.importance.zero_()

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0 and hasattr(self, "adalora_P"):
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    # AdaLoRA forward
                    x_dropout = self.lora_dropout(x)
                    after_Q = F.linear(x_dropout, self.adalora_Q.T)
                    after_E = after_Q * self.adalora_E
                    after_P = F.linear(after_E, self.adalora_P)
                    output = after_P.to(expected_dtype) * self.scaling
                    result += output
                else:
                    x_dropout = self.lora_dropout(x)
                    after_Q = F.linear(x_dropout, self.adalora_Q.T)
                    after_E = after_Q * self.adalora_E
                    after_P = F.linear(after_E, self.adalora_P)
                    output = after_P * self.scaling
                    result += output
            return result

