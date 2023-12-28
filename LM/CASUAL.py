import transformers
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
import os
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import csv
import pandas as pd


# Casual LLM for extracting the keywords from the raw text file
# facebook/opt-66b; mosaicml/mpt-30b-instruct; mosaicml/mpt-30b ; meta-llama/Llama-2-7b-hf;  meta-llama/Llama-2-70b-hf  ; tiiuae/falcon-40b-instruct ;
# Summnarization: facebook/bart-large-cnn;


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(
        description='Generate the text from the raw text attribute.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
    parser.add_argument('--model_name', type=str, default='facebook/opt-2.7b',
                        help='Name or path of the Huggingface model')
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
    parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
    parser.add_argument('--fp16', type=bool, default=True, help='if fp16')
    parser.add_argument('--cls', action='store_true', help='whether use first token to represent the whole text')

    # 加载token
    access_token = "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"
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

    output_file = Text_path + name + '_' + model_name.split('/')[-1].replace("-", "_") + ".csv"
    print(output_file)

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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=access_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)

    model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True,
                                                      token=access_token)

    # pipe = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     token=access_token,
    #     trust_remote_code=True,
    #     device_map="auto",
    # )

    Demonstration = """
The mechanistic basis of data dependence and abrupt learning in an in-context classification task. Transformer models exhibit in-context learning: the ability to accurately predict the response to a novel query based on illustrative examples in the input sequence, which contrasts with traditional in-weights learning of query-output relationships. What aspects of the training data distribution and architecture favor in-context vs in-weights learning? Recent work has shown that specific distributional properties inherent in language, such as burstiness, large dictionaries and skewed rank-frequency distributions, control the trade-off or simultaneous appearance of these two forms of learning. We first show that these results are recapitulated in a minimal attention-only network trained on a simplified dataset. In-context learning (ICL) is driven by the abrupt emergence of an induction head, which subsequently competes with in-weights learning. By identifying progress measures that precede in-context learning and targeted experiments, we construct a two-parameter model of an induction head which emulates the full data distributional dependencies displayed by the attention-based network. A phenomenological model of induction head formation traces its abrupt emergence to the sequential learning of three nested logits enabled by an intrinsic curriculum. We propose that the sharp transitions in attention-based networks arise due to a specific chain of multi-layer operations necessary to achieve ICL, which is implemented by nested nonlinearities sequentially learned during training.
Summarise the keywords from the above text.
Keywords:
in-context learning, mechanistic interpretability, language models, induction heads.

LRM: Large Reconstruction Model for Single Image to 3D. We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs including real-world in-the-wild captures and images from generative models. Video demos and interactable 3D meshes can be found on this anonymous website: https://scalei3d.github.io/LRM.
Summarise the keywords from the above text.
Keywords:
3D Reconstruction, Large-Scale, Transformers.

Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis. One-shot 3D talking portrait generation aims to reconstruct a 3D avatar from an unseen image, and then animate it with a reference video or audio to generate a talking portrait video. The existing methods fail to simultaneously achieve the goals of accurate 3D avatar reconstruction and stable talking face animation. Besides, while the existing works mainly focus on synthesizing the head part, it is also vital to generate natural torso and background segments to obtain a realistic talking portrait video. To address these limitations, we present Real3D-Potrait, a framework that (1) improves the one-shot 3D reconstruction power with a large image-to-plane model that distills 3D prior knowledge from a 3D face generative model; (2) facilitates accurate motion-conditioned animation with an efficient motion adapter; (3) synthesizes realistic video with natural torso movement and switchable background using a head-torso-background super-resolution model; and (4) supports one-shot audio-driven talking face generation with a generalizable audio-to-motion model. Extensive experiments show that Real3D-Portrait generalizes well to unseen identities and generates more realistic talking portrait videos compared to previous methods. Video samples are available at https://real3dportrait.github.io.
Summarise the keywords from the above text.
Keywords:
Neural Radiance Field, One-shot Talking Face Generation.

Entropic Neural Optimal Transport via Diffusion Processes. We propose a novel neural algorithm for the fundamental problem of computing the entropic optimal transport (EOT) plan between probability distributions which are accessible by samples. Our algorithm is based on the saddle point reformulation of the dynamic version of EOT which is known as the Schrödinger Bridge problem. In contrast to the prior methods for large-scale EOT, our algorithm is end-to-end and consists of a single learning step, has fast inference procedure, and allows handling small values of the entropy regularization coefficient which is of particular importance in some applied problems. Empirically, we show the performance of the method on several large-scale EOT tasks. The code for the ENOT solver can be found at https://github.com/ngushchin/EntropicNeuralOptimalTransport
Summarise the keywords from the above text.
Keywords:
Optimal transport, Schrödinger Bridge, Entropy regularized OT, Neural Networks, Unpaired Learning.

Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models. Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space: By adding the fine-tuned weights of different tasks, the model's performance can be improved on these tasks, while negating them leads to task forgetting. Yet, our understanding of the effectiveness of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show that fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we provide theoretical and empirical analyses of the neural tangent kernel (NTK) of these models and establish a compelling link between task arithmetic and the spatial localization of the NTK eigenfunctions. Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models through the NTK linearization.
Summarise the keywords from the above text.
Keywords:
model editing, transfer learning, neural tangent kernel, vision-language pre-training, deep learning science.
"""
    # Summary
    prompt = """
Summarise the keywords from the above text.
Keywords:
"""

    def add_prompt(example, column_name='TA'):
        example[f"{column_name}"] = f"{Demonstration}\n{example[f'{column_name}']}\n{prompt}"
        return example

    prompt_dataset = dataset.map(add_prompt)





    # for out in tqdm(pipe(KeyDataset(prompt_dataset['train'], "TA"), do_sample=True, max_new_tokens=20, use_cache=True, repetition_penalty=2,
    #                      top_k=10, num_return_sequences=3, eos_token_id=tokenizer.eos_token_id, return_full_text=False)):
    #     print(out)

    # 打开CSV文件并创建写入器
    generated_text_list = []  # 创建一个列表用于存储生成的文本

    for t in tqdm(range(len(KeyDataset(prompt_dataset['train'], "TA")))):
        inputs = tokenizer(t, return_tensors="pt").to("cuda")
        generated_ids = model_8bit.generate(**inputs)
        out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, do_sample=True, max_new_tokens=20,
                                     use_cache=True, repetition_penalty=2.5,
                                     top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,
                                     return_full_text=False)

        generated_text = out[0]['generated_text']
        generated_text_list.append(generated_text)

    df = pd.DataFrame({'Keywords': generated_text_list})
    df.to_csv(output_file, index=False)

    # Pipe 的方式生成并保存文本
    # for out in tqdm(pipe(KeyDataset(prompt_dataset['train'], "TA"), do_sample=True, max_new_tokens=20, use_cache=True,
    #                      repetition_penalty=2.5,
    #                      top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,
    #                      return_full_text=False)):
    #     generated_text = out[0]['generated_text']
    #     generated_text_list.append(generated_text)
    #
    # df = pd.DataFrame({'Keywords': generated_text_list})
    # df.to_csv(output_file, index=False)

    print("CSV file has been generated successfully.")


if __name__ == "__main__":
    main()
"""
python CASUAL.py --csv_file /dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/ogbn_arxiv.csv
"""
