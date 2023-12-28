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

    Demonstration = """The mechanistic basis of data dependence and abrupt learning in an in-context classification task. Transformer models exhibit in-context learning: the ability to accurately predict the response to a novel query based on illustrative examples in the input sequence, which contrasts with traditional in-weights learning of query-output relationships. What aspects of the training data distribution and architecture favor in-context vs in-weights learning? Recent work has shown that specific distributional properties inherent in language, such as burstiness, large dictionaries and skewed rank-frequency distributions, control the trade-off or simultaneous appearance of these two forms of learning. We first show that these results are recapitulated in a minimal attention-only network trained on a simplified dataset. In-context learning (ICL) is driven by the abrupt emergence of an induction head, which subsequently competes with in-weights learning. By identifying progress measures that precede in-context learning and targeted experiments, we construct a two-parameter model of an induction head which emulates the full data distributional dependencies displayed by the attention-based network. A phenomenological model of induction head formation traces its abrupt emergence to the sequential learning of three nested logits enabled by an intrinsic curriculum. We propose that the sharp transitions in attention-based networks arise due to a specific chain of multi-layer operations necessary to achieve ICL, which is implemented by nested nonlinearities sequentially learned during training.
Summarise the keywords from the above text.
Keywords:
in-context learning, mechanistic interpretability, language models, induction heads.
"""
    # Summary
    prompt = """Summarise the keywords from the above text.\nKeywords:
"""

    def add_prompt(example, column_name='abstract'):
        example[f"{column_name}"] = f"{Demonstration}\n\n{example[f'{column_name}']}\n\n{prompt}"
        return example

    prompt_dataset = dataset.map(add_prompt)

    for out in tqdm(pipe(KeyDataset(prompt_dataset['train'], "abstract"), do_sample=True, max_new_tokens=20,
                         top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, return_full_text=False)):
        print(out)



if __name__ == "__main__":
    main()





