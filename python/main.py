import os
import time

import torch
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import BloomConfig, BloomForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

model_name = 'NousResearch/Llama-2-7b-hf'

world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
#  tokenizer = LlamaTokenizer.from_pretrained(model_name, config=config)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.half,
        low_cpu_mem_usage=True,
)

## ds-inference
infer_config = dict(
        tensor_parallel={'tp_size': world_size},
        dtype=torch.half,
        replace_with_kernel_inject=True,
)
model = deepspeed.init_inference(model, config=infer_config)
model.eval()


prompt = '''Explain the concept of cylindricity to a second-grade student:
'''

for _ in range(50):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    gen_tokens = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.5,
        max_new_tokens=300,
    )
    gen_text = tokenizer.decode(gen_tokens[0])
    print(gen_text)
