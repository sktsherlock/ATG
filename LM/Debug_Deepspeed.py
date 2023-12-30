import os
import time

import torch
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


access_token = "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"
#  model_name = 'bigscience/bloomz-7b1-mt'
model_name = 'meta-llama/Llama-2-70b-hf'


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
#  tokenizer = LlamaTokenizer.from_pretrained(model_name, config=config)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.half,
        low_cpu_mem_usage=True,
        token=access_token,
)

## ds-inference
infer_config = dict(
        tensor_parallel={'tp_size': world_size},
        dtype=torch.half,
        replace_with_kernel_inject=True,
)
model = deepspeed.init_inference(model, config=infer_config)
model.eval()


prompt = '''Summarize this for a second-grade student:

Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.'''

for _ in range(50):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    gen_tokens = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.5,
        max_new_tokens=20,
    )
    gen_text = tokenizer.decode(gen_tokens[0])
    print(gen_text)