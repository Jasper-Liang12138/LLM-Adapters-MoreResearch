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
"""
QLoRA: Efficient Finetuning of Quantized LLMs

QLoRA uses 4-bit quantization to reduce memory footprint while maintaining
the same performance as full fine-tuning. The base model is quantized to 4-bit
using bitsandbytes, while LoRA adapters remain in full precision.

To use QLoRA, load your model with 4-bit quantization:
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Then apply QLoRA
    from peft import QLoRAConfig, get_peft_model
    config = QLoRAConfig(...)
    model = get_peft_model(model, config)
"""
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

# Import LoRA classes - QLoRA is essentially LoRA with 4-bit quantization
from .lora import (
    LoraConfig, 
    LoraModel, 
    LoraLayer, 
    Linear, 
    MergedLinear, 
    mark_only_lora_as_trainable,
)

# Import 8-bit classes conditionally
try:
    from .lora import Linear8bitLt, MergedLinear8bitLt
except ImportError:
    Linear8bitLt = None
    MergedLinear8bitLt = None

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


# Linear4bit LoRA classes for QLoRA
Linear4bit = None
MergedLinear4bit = None

if is_bnb_available() and hasattr(bnb.nn, "Linear4bit"):
    class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
        """LoRA implemented in a 4-bit quantized linear layer."""
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                quant_type=kwargs.get("quant_type", "nf4"),
                quant_storage=kwargs.get("quant_storage", torch.uint8),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                    result += output
                else:
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                    result += output
            return result

    class MergedLinear4bit(bnb.nn.Linear4bit, LoraLayer):
        """LoRA implemented in a 4-bit quantized linear layer with merged weights."""
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                quant_type=kwargs.get("quant_type", "nf4"),
                quant_storage=kwargs.get("quant_storage", torch.uint8),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            if out_features % len(enable_lora) != 0:
                raise ValueError("The length of enable_lora must divide out_features")
            self.enable_lora = enable_lora
            # Actual trainable parameters
            if r > 0 and any(enable_lora):
                self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
                self.lora_B = nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                )
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                # Compute the indices
                self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[enable_lora, :] = True
                self.lora_ind = self.lora_ind.view(-1)
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def zero_pad(self, x):
            result = x.new_zeros((*x.shape[:-1], self.out_features))
            result = result.view(-1, self.out_features)
            result[:, self.lora_ind] = x.reshape(
                -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
            )
            return result.view((*x.shape[:-1], self.out_features))

        def forward(self, x: torch.Tensor):
            result = super().forward(x)
            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                    result += output
                else:
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B) * self.scaling
                    result += output
            return result


@dataclass
class QLoRAConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.QLoRA`].
    
    QLoRA is essentially LoRA with 4-bit quantization. The base model should be
    loaded with 4-bit quantization using BitsAndBytesConfig before applying QLoRA.
    
    Args:
        r (`int`): LoRA attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply LoRA to.
        lora_alpha (`float`): The alpha parameter for LoRA scaling.
        lora_dropout (`float`): The dropout probability for LoRA layers.
        merge_weights (`bool`):
            Whether to merge the weights of the LoRA layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    def __post_init__(self):
        self.peft_type = PeftType.QLORA
        # Warn if bitsandbytes is not available
        if not is_bnb_available():
            warnings.warn(
                "bitsandbytes is not available. QLoRA requires 4-bit quantization "
                "which needs bitsandbytes. Please install it with: pip install bitsandbytes"
            )


class QLoRAModel(LoraModel):
    """
    Creates Quantized Low Rank Adapter (QLoRA) model from a pretrained transformers model.
    
    QLoRA uses 4-bit quantization for the base model while keeping LoRA adapters
    in full precision. The model should be loaded with 4-bit quantization before
    applying QLoRA.
    
    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted (should be 4-bit quantized).
        config ([`QLoRAConfig`]): The configuration of the QLoRA model.
    
    Returns:
        `torch.nn.Module`: The QLoRA model.
    
    Example::
    
        >>> from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        >>> from peft import QLoRAConfig, get_peft_model
        >>> 
        >>> # Load model with 4-bit quantization
        >>> bnb_config = BitsAndBytesConfig(
        ...     load_in_4bit=True,
        ...     bnb_4bit_quant_type="nf4",
        ...     bnb_4bit_use_double_quant=True,
        ...     bnb_4bit_compute_dtype=torch.bfloat16
        ... )
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "model_name",
        ...     quantization_config=bnb_config,
        ...     device_map="auto"
        ... )
        >>> 
        >>> # Apply QLoRA
        >>> config = QLoRAConfig(
        ...     r=64,
        ...     lora_alpha=16,
        ...     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ...     lora_dropout=0.1,
        ...     bias="none",
        ...     task_type="CAUSAL_LM",
        ... )
        >>> model = get_peft_model(model, config)
    
    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The quantized model to be adapted.
        - **peft_config** ([`QLoRAConfig`]): The configuration of the QLoRA model.
    """

    def __init__(self, config, model):
        # Check if model is quantized
        loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False) #getattr(object, attribute_name, default) ，获取 object 的属性 attribute_name，如果不存在，则返回 default。
        loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
        
        if not loaded_in_4bit and not loaded_in_8bit:
            warnings.warn(
                "The model does not appear to be quantized. QLoRA is designed to work with "
                "4-bit quantized models. Consider loading the model with 4-bit quantization:\n"
                "from transformers import BitsAndBytesConfig\n"
                "bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)\n"
                "model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)"
            )
        
        if loaded_in_4bit and not is_bnb_available():
            raise ImportError(
                "To use QLoRA with 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        
        # QLoRA is essentially LoRA, so we can use the parent class
        super().__init__(config, model)
        
        # Store quantization info
        self.is_loaded_in_4bit = loaded_in_4bit # 是否加载了4位量化（bool）
        self.is_loaded_in_8bit = loaded_in_8bit # 是否加载了8位量化（bool）

    def _find_and_replace(self):
        # Check for quantization
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        
        if loaded_in_4bit and not is_bnb_available():
            raise ImportError(
                "To use QLoRA with 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
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
                
                # Handle 4-bit quantized layers
                if loaded_in_4bit and is_bnb_available() and hasattr(bnb.nn, "Linear4bit") and isinstance(target, bnb.nn.Linear4bit):
                    # Check if Linear4bit classes are defined
                    if Linear4bit is None or MergedLinear4bit is None:
                        raise ImportError(
                            "Linear4bit LoRA classes are not available. "
                            "This usually means bitsandbytes is not properly installed or "
                            "the version does not support Linear4bit."
                        )
                    # Extract quantization parameters from the original layer
                    kwargs.update({
                        "compute_dtype": getattr(target, "compute_dtype", torch.float32),
                        "quant_type": getattr(target, "quant_type", "nf4"),
                        "quant_storage": getattr(target, "quant_storage", torch.uint8),
                    })
                    if self.peft_config.enable_lora is None:
                        new_module = Linear4bit(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear4bit(target.in_features, target.out_features, bias=bias, **kwargs)
                # Handle 8-bit quantized layers
                elif loaded_in_8bit and is_bnb_available() and isinstance(target, bnb.nn.Linear8bitLt):
                    if Linear8bitLt is None or MergedLinear8bitLt is None:
                        raise ImportError(
                            "Linear8bitLt classes are not available. "
                            "This usually means bitsandbytes is not properly installed."
                        )
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                # Handle standard Linear layers
                elif isinstance(target, nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                # Handle MergedLinear cases
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                else:
                    # Skip if we can't determine the layer type
                    continue
                    
                self._replace_module(parent, target_name, new_module, target)
                
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

