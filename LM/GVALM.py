import logging
import os
import dgl
import sys
import shutil
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import List, Optional
from types import SimpleNamespace as SN
import datasets
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, Dataset
import torch.nn.functional as F

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    AutoConfig,
    HfArgumentParser,
    set_seed,
)

from Task import CLSClassifier, DualClassifier
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

check_min_version("4.35.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def split_graph(nodes_num, train_ratio, val_ratio, labels, fewshots=None):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    if fewshots is not None:
        train_ids = []

        unique_labels = np.unique(labels)  # 获取唯一的类别标签
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]  # 获取属于当前类别的样本索引
            np.random.shuffle(label_indices)  # 对当前类别的样本索引进行随机排序

            fewshot_indices = label_indices[:fewshots]  # 选择指定数量的few-shot样本
            train_ids.extend(fewshot_indices)

        remaining_indices = np.setdiff1d(indices, train_ids)  # 获取剩余的样本索引
        np.random.shuffle(remaining_indices)  # 对剩余样本索引进行随机排序

        val_size = int(len(remaining_indices) * val_ratio)  # 计算验证集大小

        val_ids = remaining_indices[:val_size]  # 划分验证集
        test_ids = remaining_indices[val_size:]  # 划分测试集

    else:

        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_ids = indices[:train_size]
        val_ids = indices[train_size:train_size + val_size]
        test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


class Sequence:
    def __init__(self, cf):
        self.cf = cf
        self.ndata = {}
        self.n_nodes = cf['n_nodes']
        self.max_length = cf['max_length']
        self.inductive = False
        self.graph_path = cf['graph_path']
        self.token_folder = cf['token_folder']
        self.train_ratio = cf['train_ratio']
        self.val_ratio = cf['val_ratio']
        self.fewshots = cf['fewshots']

        self.info = {
            'input_ids': SN(shape=(self.n_nodes, self.max_length), type=np.uint16),
            'attention_mask': SN(shape=(self.n_nodes, self.max_length), type=bool),
        }
        for k, info in self.info.items():
            info.path = f'{self.token_folder}{k}.npy'

    def init(self):
        # 加载token相关信息
        self._load_data_fields()
        # 加载图相关信息，如节点标签，数据集划分
        g = dgl.load_graphs(self.graph_path)[0][0]
        labels = g.ndata['label']
        train_idx, val_idx, test_idx = split_graph(g.num_nodes(), self.train_ratio, self.val_ratio, labels,
                                                   fewshots=self.fewshots)
        # 转为无向图后 获取邻居
        g = dgl.to_bidirected(g)

        self.neighbours = list(g.adjacency_matrix_scipy().tolil().rows)

        self.train_x = train_idx
        self.val_x = val_idx
        self.test_x = test_idx
        self.num_labels = (labels.max() + 1).item()

        self.ndata['labels'] = labels

        if self.inductive:
            self.edge_index = self.get_train_edge()

        return self


    def node_label(self, node_id):
        labels = self.ndata['labels'][node_id]
        return F.one_hot(labels, num_classes=self.num_labels).type(torch.FloatTensor)

    def _load_data_fields(self):
        for k in self.info:
            i = self.info[k]
            try:
                self.ndata[k] = np.load(i.path,
                                        allow_pickle=True)  # np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
            except:
                raise ValueError(f'Shape not match {i.shape}')

    def get_tokens(self, node_id):
        _load = lambda k: torch.IntTensor(np.array(self.ndata[k][node_id]))
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = torch.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        return item

    def get_nb_tokens(self, item, node_id):
        _load = lambda k: torch.IntTensor(np.array(self.ndata[k][node_id]))
        item['nb_input_ids'] = torch.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        item['nb_attention_mask'] = _load('attention_mask')
        return item

    def __getitem__(self, k):
        return self.ndata[k]


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data: Sequence, topology: bool):
        super().__init__()
        self.d = data
        self.topology = topology

    def __getitem__(self, node_id):
        item = self.d.get_tokens(node_id)
        item['labels'] = self.d.node_label(node_id)
        if self.topology:
            neighbours = self.d.neighbours[node_id]
            if neighbours:
                k = np.random.choice(neighbours, 1)[0]
                item = self.d.get_nb_tokens(item, k)
            else:
                # 防止孤立点报错
                k = node_id
                item = self.d.get_nb_tokens(item, k)
        # item['labels'] = F.one_hot(torch.from_numpy(self.d.ndata['labels'][node_id]), num_classes=self.d.num_labels).type(torch.FloatTensor)
        return item

    def __len__(self):
        return self.d.n_nodes


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    csv_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the CSV File."}
    )
    data_name: Optional[str] = field(
        default=None, metadata={"help": "The dataname to be used for splitting dataaset. like ogbn-arxiv"}
    )
    graph_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the graph file."}
    )
    text_column_names: Optional[str] = field(
        default='text', metadata={"help": "Name of the column containing the text attribute."}
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence/document length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    train_ratio: float = field(
        default=0.6,
        metadata={
            "help": "The ratio of training"
        },
    )
    val_ratio: float = field(
        default=0.2,
        metadata={
            "help": "The ratio of validation"
        },
    )
    fewshots: int = field(
        default=None,
        metadata={
            "help": (
                "The fewshots number; if None, dont use fewshots"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    token_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the token file."}
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
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
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
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='prajjwal1/bert-tiny',
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
    filename: Optional[str] = field(
        default=None,
        metadata={"help": "Where you save the adapter model"},
    )
    out_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the training processes"},
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tuning language models"},
    )
    topology: bool = field(
        default=False,
        metadata={"help": "Whether to use the neighbor information for node classification."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    training_objective: str = field(
        default="CLS",
        metadata={"help": "The training objective of PLM to use (CLS or MEAN) for finetuning downstream model."}
    )
    drop_out: float = field(
        default=0.2,
        metadata={"help": "The drop out ratio"}
    )
    resduial: bool = field(
        default=True,
        metadata={"help": "Whether to ues skip connection."}
    )
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "The label smoothing factor to use"}
    )
    cls_head_bias: bool = field(
        default=True,
        metadata={"help": "Whether to add bias for classifier head."}
    )
    unfreeze_layers: int = field(
        default=None,
        metadata={
            "help": "The layers to unfreeze"
        },
    )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = training_args.output_dir + model_args.model_name_or_path.split('/')[-1].replace("-",
                                                                                                               "_") + '/' + f't_{data_args.train_ratio}_v_{data_args.val_ratio}_d_{model_args.drop_out}_w_{training_args.warmup_ratio}_lr_{training_args.learning_rate}_e_{training_args.num_train_epochs}_b_{training_args.per_device_train_batch_size}_u{model_args.unfreeze_layers}_s_{training_args.seed}'

    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 加载数据集
    # Loading a dataset from your local files. CSV training and evaluation files are needed.

    csv_file = data_args.csv_file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))

    data_files = os.path.join(base_dir, csv_file)

    # 读取CSV文件
    df = pd.read_csv(data_files)
    text_data = df[data_args.text_column_names].tolist()

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config.cls_head_bias = model_args.cls_head_bias
    config.problem_type = "single_label_classification"
    logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    token_folder = data_args.token_folder + model_args.model_name_or_path.split('/')[-1].replace("-", "_") + '/' + f'len_{max_seq_length}/'
    if not os.path.exists(token_folder):
        print(f'The token folder {token_folder} does not exist')
        os.makedirs(token_folder)

    if os.path.isdir(token_folder):
        file_list = os.listdir(token_folder)
        if len(file_list) != 3:
            # 编码文本数据并转为数据集
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenized = tokenizer(text_data, padding=True, truncation=True, max_length=max_seq_length, return_tensors='pt')
            dataset = Dataset.from_dict(tokenized)
            print(dataset)

            for k in tokenized.data:
                with open(os.path.join(token_folder, f'{k}.npy'), 'wb') as f:
                    np.save(f, tokenized.data[k])
        else:
            pass
    # 将本地token file读入到数据集中
    cf = {'n_nodes': len(df), 'max_length': max_seq_length, 'graph_path': data_args.graph_path,
          'token_folder': token_folder, 'train_ratio': data_args.train_ratio, 'val_ratio': data_args.val_ratio, 'fewshots': data_args.fewshots}

    # 创建数据集 Sequence
    d = Sequence(cf).init()
    full_data = SeqDataset(d, topology=model_args.topology)

    subset_data = lambda sub_idx: torch.utils.data.Subset(full_data, sub_idx)
    Data = {_: subset_data(getattr(d, f'{_}_x'))
                     for _ in ['train', 'val', 'test']}
    if data_args.shuffle_train_dataset:
        logger.info("Shuffling the training dataset")
        print(f'Before shuffling the training dataset:{d.train_x}')
        np.random.shuffle(d.train_x)
        print(f'After shuffling the training dataset:{d.train_x}')
        Data['train'] = subset_data(d.train_x)

    train_data = Data['train']
    eval_dataset = Data['val']
    predict_dataset = Data['test']
    num_labels = (d.ndata['labels'].max() + 1).item()

    # 创建用于分类的模型
    encoder = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model_args.unfreeze_layers is not None:
        for param in encoder.parameters():
            param.requires_grad = False

        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        assert trainable_params == 0
        for param in encoder.encoder.layer[-model_args.unfreeze_layers:].parameters():
            param.requires_grad = True

        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        print(f" Pass the freeze layer, the LM Encoder  parameters are {trainable_params}")

    if model_args.training_objective == "CLS":
        model = CLSClassifier(
            encoder, num_labels,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean')
        )
    elif model_args.training_objective == 'Mean':
        model = DualClassifier(
            encoder, num_labels,
            inputs_dim=data_args.image_dim,
            dropout=model_args.drop_out,
            mode=model_args.mode,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean'),

        )
    else:
        raise ValueError("Training objective should be either CLS or Mean.")


    if data_args.metric_name is not None:
        if data_args.metric_name == 'f1':
            metric = (evaluate.load(data_args.metric_name, average='macro'))
        else:
            metric = (evaluate.load(data_args.metric_name))
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        metric = evaluate.load("accuracy")
        logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids.argmax(1),
                                average='macro') if data_args.metric_name == 'f1' else metric.compute(predictions=preds,
                                                                                                      references=p.label_ids.argmax(1))
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        metrics = trainer.evaluate(eval_dataset=predict_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    shutil.rmtree(training_args.output_dir)


if __name__ == "__main__":
    main()
