import transformers
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from datasets import load_dataset
import os
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
# Casual LLM for extracting the keywords from the raw text file
# facebook/opt-66b; mosaicml/mpt-30b-instruct; mosaicml/mpt-30b ; meta-llama/Llama-2-7b-hf;  meta-llama/Llama-2-70b-hf  ; tiiuae/falcon-40b-instruct ;
# Summnarization: facebook/bart-large-cnn;


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(
        description='Generate the text from the raw text attribute.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
    parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny',
                        help='Name or path of the Huggingface model')
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
    parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
    parser.add_argument('--fp16', type=bool, default=True, help='if fp16')
    parser.add_argument('--cls', action='store_true', help='whether use first token to represent the whole text')


    # 解析命令行参数
    args = parser.parse_args()
    model_name = args.model_name
    name = args.name

    tokenizer_name = args.tokenizer_name

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))
    Text_path = os.path.join(base_dir, args.path)
    cache_path = f"{Text_path}cache/"

    if not os.path.exists(Text_path):
        os.makedirs(Text_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    output_file = Text_path + name + '_' + model_name.split('/')[-1].replace("-", "_")

    # Set seed before initializing model.
    set_seed(args.seed)

    # 加载数据集
    # Loading a dataset from your local files. CSV training and evaluation files are needed.
    csv_file = args.csv_file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))

    data_files = os.path.join(base_dir, csv_file)

    dataset = load_dataset(
        "csv",
        data_files=data_files,
    )

    # 加载模型和分词器
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )


    # Summary
    prompt = """Summarise the keywords from the above text. For example: Neural Radiance Field, One-shot Talking Face Generation.
                Keywords:
            """

    def add_prompt(example, column_name='text'):
        example[f"{column_name}"] = f"{example[f'{column_name}']}\n\n{prompt}"
        return example

    prompt_dataset = dataset.map(add_prompt)

    for out in tqdm(pipe(KeyDataset(prompt_dataset, "text"), do_sample=True, max_new_tokens=20,
                         top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, return_full_text=False)):
        print(out)


if __name__ == "__main__":
    main()





