import torch
import torch_npu
import deepspeed
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '23456'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
dist.init_process_group(backend='hccl')
print('torch dist world_size:', dist.get_world_size())
import deepspeed.comm as dc
print('deepspeed comm world_size:', dc.get_world_size())
dist.destroy_process_group()

import transformers, inspect
src = inspect.getsource(transformers.modeling_utils.PreTrainedModel.from_pretrained)
for i, line in enumerate(src.split('\n')):
    if 'zero' in line.lower() or 'deepspeed' in line.lower() or 'meta' in line.lower():
        print(i, line)
