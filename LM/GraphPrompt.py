from transformers import pipeline, AutoTokenizer, set_seed
import torch
from datasets import load_dataset
import os
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import pandas as pd
import deepspeed
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Path to the config file')
parser.add_argument('--num', type=int, default=1, help='The prompt numbers')
args = parser.parse_args()

# 加载token
access_token = "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"

world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

# 解析命令行参数
model_name = config.model_name

tokenizer_name = config.tokenizer_name

Text_path = config.path

if not os.path.exists(Text_path):
    os.makedirs(Text_path)

output_file = Text_path + 'Keywords_' + model_name.split('/')[-1].replace("-", "_") + f"_{args.num}_shot.csv"
print(output_file)

# Set seed before initializing model.
set_seed(config.seed)

# 加载数据集
# Loading a dataset from your local files. CSV training and evaluation files are needed.
csv_file = config.csv_file
root_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(root_dir.rstrip('/'))

data_files = os.path.join(base_dir, csv_file)

dataset = load_dataset(
    "csv",
    data_files=data_files,
)

# 加载模型和分词器
if tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=access_token)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)

pipe = pipeline(
    config.task_name,
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    token=access_token,
    trust_remote_code=True,
    device_map="auto",
)

if config.speed:
    pipe.model = deepspeed.init_inference(pipe.model,
                                          max_out_tokens=4096,
                                          tensor_parallel={'tp_size': world_size},
                                          dtype=torch.half,
                                          replace_with_kernel_inject=True)

pipe.model.eval()


