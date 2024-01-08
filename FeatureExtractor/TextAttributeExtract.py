import argparse
import numpy as np
import pandas as pd
import torch
import os
import sys
from typing import Optional
from transformers import AutoTokenizer, AutoModel, TrainingArguments, PreTrainedModel, Trainer, DataCollatorWithPadding, \
    AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset, load_dataset
from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is not None:
            pass
        else:
            test_extension = self.test_file.split(".")[-1]
            assert test_extension in ["csv", "json"], "`test_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    training_objective: str = field(
        default="CLS",
        metadata={"help": "The training objective of bloom to use (CLS or CLM) for finetuning downstream model."}
    )
    lm_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "If greater than 0.0, CLM loss will be computed on all positions, otherwise only on the last position."},
    )
    template_config: str = field(
        default="template_pool.json",
        metadata={"help": "Where to load the predefined template."}
    )
    template_name: str = field(
        default="template_1",
        metadata={"help": "Which template to use."}
    )
    template: str = field(
        default="Does the query \"{}\" match the keyword \"{}\"? Answer:",
        metadata={"help": "The template (prompt) to tune downstream QK task using CLM objective."}
    )
    truncate_in_pairs: bool = field(
        default=False,
        metadata={"help": "Whether to truncate in pairs before combination with prompt, only work for CLM mode."},
    )
    use_features: bool = field(
        default=True,
        metadata={"help": "Whether to use data features or not."},
    )
    feature_separator: str = field(
        default=" ##! ",
        metadata={"help": "Customize separator for concatenating features."}
    )
    positive_token: int = field(
        default=31559,
        metadata={"help": "The id of positive token, default 'Yes'."},
    )
    negative_token: int = field(
        default=4309,
        metadata={"help": "The id of negative token, default 'No'."},
    )
    cls_head_tunable: bool = field(
        default=True,
        metadata={"help": "Whether to tune the classifier head."},
    )
    cls_head_bias: bool = field(
        default=False,
        metadata={"help": "Whether to add bias for classifier head."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA or not."},
    )
    save_lora_with_prefix: bool = field(
        default=False,
        metadata={"help": "Whether to keep the prefix ('transformer') when saving lora weights."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The rank of LoRA."},
    )
    lora_train_bias: str = field(
        default="none",
        metadata={"help": "Whether to train bias, choices: none, lora_only and all."},
    )
    lora_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha of LoRA when adding the LoRA parameters to the origin ones."},
    )
    lora_param: str = field(
        default="Q.V",
        metadata={
            "help": "The parameter groups to apply LoRA, including E (embeddings), Q (attn query), K (attn key), "
                    "V (attn value), O (attn output) and F (feedforward), splitted by dot, e.g. Q.V means applying "
                    "LoRA to Q and V."
        }
    )
    lora_ckpt: str = field(
        default=None,
        metadata={"help": "The checkpoint path of LoRA checkpoint."},
    )
    lora_ckpt_old_format: bool = field(
        default=False,
        metadata={"help": "Whether the LoRA checkpoint is in old format."},
    )
    lora_layers: int = field(
        default=-1,
        metadata={"help": "The number of top layers to apply LoRA. Set to -1 to apply LoRA to all layers."},
    )
    random_init: bool = field(
        default=False,
        metadata={"help": "If true, do not load the pretained weights and initialize the model with random weights."},
    )


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(
        description='Process text data and save the overall representation as an NPY file.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
    parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny',
                        help='Name or path of the Huggingface model')
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
    parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
    parser.add_argument('--fp16', type=bool, default=True, help='if fp16')
    parser.add_argument('--cls', action='store_true', help='whether use first token to represent the whole text')

    # 解析命令行参数
    args = parser.parse_args()
    csv_file = args.csv_file
    text_column = args.text_column
    model_name = args.model_name
    name = args.name
    max_length = args.max_length
    batch_size = args.batch_size
    access_token = "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"

    tokenizer_name = args.tokenizer_name

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))
    Feature_path = os.path.join(base_dir, args.path)
    cache_path = f"{Feature_path}cache/"
    print(Feature_path)
    print(model_name)

    if not os.path.exists(Feature_path):
        os.makedirs(Feature_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    output_file = Feature_path + name + '_' + model_name.split('/')[-1].replace("-", "_") + '_' + str(max_length)

    class CLSEmbInfModel(PreTrainedModel):
        def __init__(self, model):
            super().__init__(model.config)
            self.encoder = model

        @torch.no_grad()
        def forward(self, input_ids, attention_mask):
            # Extract outputs from the model3
            outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
            # Use CLS Emb as sentence emb.
            # Use  pooler_output ? Dont use pooler_output
            #node_cls_emb = outputs.pooler_output
            node_cls_emb = outputs.last_hidden_state[:, 0, :]  # Last layer
            return TokenClassifierOutput(logits=node_cls_emb)

    class MeanEmbInfModel(PreTrainedModel):
        def __init__(self, model):
            super().__init__(model.config)
            self.encoder = model

        @torch.no_grad()
        def forward(self, input_ids, attention_mask):
            # Extract outputs from the model
            outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
            # Use Mean Emb as sentence emb.
            node_mean_emb = torch.mean(outputs.last_hidden_state, dim=1)
            return TokenClassifierOutput(logits=node_mean_emb)

    # 读取CSV文件
    df = pd.read_csv(os.path.join(base_dir, csv_file))
    text_data = df[text_column].tolist()

    # 加载模型和分词器
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=access_token,  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token,  trust_remote_code=True)

    # 编码文本数据并转为数据集
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_inputs = tokenizer(text_data, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    dataset = Dataset.from_dict(encoded_inputs)

    model = AutoModel.from_pretrained(model_name,  trust_remote_code=True, token=access_token)

    CLS_Feateres_Extractor = CLSEmbInfModel(model)
    Mean_Features_Extractor = MeanEmbInfModel(model)
    CLS_Feateres_Extractor.eval()
    Mean_Features_Extractor.eval()

    inference_args = TrainingArguments(
        output_dir=cache_path,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        fp16_full_eval=args.fp16,
    )

    # CLS 特征提取
    if args.cls:
        if not os.path.exists(output_file + "_cls.npy"):
            trainer = Trainer(model=CLS_Feateres_Extractor, args=inference_args)
            cls_emb = trainer.predict(dataset)
            # 保存CLS(首个字符的表示)表示为NPY文件
            np.save(output_file + "_cls.npy", cls_emb.predictions)
        else:
            print('Existing saved CLS')

    if not os.path.exists(output_file + "_mean.npy"):
        trainer = Trainer(model=Mean_Features_Extractor, args=inference_args)
        mean_emb = trainer.predict(dataset)
        # 保存平均特征表示为NPY文件
        np.save(output_file + "_mean.npy", mean_emb.predictions)
    else:
        print('Existing saved MEAN')


if __name__ == "__main__":
    main()
