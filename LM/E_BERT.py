import logging
import os
import random
import sys
import shutil
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ogb.nodeproppred import DglNodePropPredDataset
import evaluate
import numpy as np
from datasets import Value, load_dataset
import torch
from datasets import DatasetDict, Dataset
from peft import PeftModelForFeatureExtraction, get_peft_config

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    PreTrainedModel,
    Trainer,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    AutoConfig,
    HfArgumentParser,
    set_seed,
)

from Task import CLSClassifier, MEANClassifier, AdapterClassifier
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

check_min_version("4.35.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)



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
        default='prajjwal1/bert-tiny', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "The label smoothing factor to use"}
    )
    cls_head_bias: bool = field(
        default=True,
        metadata={"help": "Whether to add bias for classifier head."}
    )
    peft_type: str = field(
        default="LORA",
        metadata={"help": "Which PEFT model to be used."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The rank of LoRA."},
    )
    lora_train_bias: str = field(
        default="none",
        metadata={"help": "Whether to train bias, choices: none, lora_only and all."},
    )
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )


class Classifier(nn.Module):
    def __init__(self, model, in_feats, n_labels):
        super().__init__()
        self.Adapter = model
        hidden_dim = in_feats
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def reset_parameters(self):
        self.Adapter.reset_parameters()

        self.classifier.reset_parameters()

    def forward(self, feat):
        # Extract outputs from the model
        outputs, feat = self.Adapter(feat)
        outputs = outputs + feat
        logits = self.classifier(outputs)
        return logits



class MLP(nn.Module):
    def __init__(
            self,
            in_feats,
            n_layers,
            n_hidden,
            activation,
            dropout=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else in_feats

            self.linears.append(nn.Linear(in_hidden, out_hidden))

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, feat):
        h = feat

        for i in range(self.n_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)

        return self.linears[-1](h), feat


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def split_dataset(nodes_num, train_ratio, val_ratio, data_name=None):
    if data_name == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=data_name)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
    else:
        np.random.seed(42)
        indices = np.random.permutation(nodes_num)

        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)
        test_idx = torch.tensor(test_idx)

    return train_idx, val_idx, test_idx


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



def set_peft_config(modeling_args):
    if modeling_args.model_name_or_path in {"facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"}:
        config = {'peft_type': modeling_args.peft_type, 'target_modules': ["q_proj", "v_proj"],
                  'r': modeling_args.lora_rank, 'bias': modeling_args.lora_train_bias,
                  'lora_alpha': modeling_args.lora_alpha, 'lora_dropout': modeling_args.lora_dropout}
    else:
        config = {'peft_type': modeling_args.peft_type, 'target_modules': ["query", "key"],
                  'r': modeling_args.lora_rank, 'bias': modeling_args.lora_train_bias,
                  'lora_alpha': modeling_args.lora_alpha, 'lora_dropout': modeling_args.lora_dropout}
    peft_config = get_peft_config(config)
    return peft_config


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = training_args.output_dir + model_args.model_name_or_path.split('/')[-1].replace("-", "_") + '/' + f't_{data_args.train_ratio}_v_{data_args.val_ratio}_d_{model_args.drop_out}_w_{training_args.warmup_ratio}_lr_{training_args.learning_rate}_e_{training_args.num_train_epochs}_b_{training_args.per_device_train_batch_size}_{model_args.peft_type}_{model_args.lora_rank}_{model_args.lora_alpha}_{model_args.lora_dropout}'
    training_args.evaluation_strategy = 'epoch'
    training_args.save_strategy = 'epoch'
    training_args.load_best_model_at_end = True
    training_args.save_total_limit = None
    training_args.do_train = True
    training_args.do_eval = True
    training_args.do_predict = True

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 加载数据集
    # Loading a dataset from your local files. CSV training and evaluation files are needed.

    csv_file = data_args.csv_file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))

    data_files = os.path.join(base_dir, csv_file)

    raw_data = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )
    train_data = raw_data['train']
    nodes_num = len(raw_data['train'])

    train_ids, val_ids, test_ids = split_dataset(nodes_num, data_args.train_ratio, data_args.val_ratio, data_name=data_args.data_name)
    # 根据划分的索引创建划分后的数据集
    train_dataset = train_data.select(train_ids)
    val_dataset = train_data.select(val_ids)
    test_dataset = train_data.select(test_ids)

    # 创建包含划分后数据集的DatasetDict对象
    raw_datasets = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    print(raw_datasets)
    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")



    label_list = get_label_list(raw_datasets, split="train")
    for split in ["validation", "test"]:
        if split in raw_datasets:
            val_or_test_labels = get_label_list(raw_datasets, split=split)
            diff = set(val_or_test_labels).difference(set(label_list))
            if len(diff) > 0:
                # add the labels that appear in val/test but not in train, throw a warning
                logger.warning(
                    f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                )
                label_list += list(diff)
    # if label is -1, we throw a warning and remove it from the label list
    for label in label_list:
        if label == -1:
            logger.warning("Label -1 found in label list, removing it.")
            label_list.remove(label)

    label_list.sort()
    num_labels = len(label_list)
    if num_labels <= 1:
        raise ValueError("You need more than one label to do classification.")


    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = set_peft_config(model_args)
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


    # 创建用于分类的模型
    # 加载PLM 作为Encoder
    encoder = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    peft_encoder = PeftModelForFeatureExtraction(encoder, config)
    peft_encoder.print_trainable_parameters()

    if model_args.training_objective == "CLS":
        model = CLSClassifier(
            peft_encoder, num_labels,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean')
        )
    elif model_args.training_objective == 'Mean':
        model = MEANClassifier(
            peft_encoder, num_labels,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean')
        )
    elif model_args.training_objective == 'Adapter':

        adapter = torch.load(model_args.filename)
        model = AdapterClassifier(
            peft_encoder, adapter=adapter,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean')
        )
    else:
        raise ValueError("Training objective should be either CLS or Mean.")
    print_trainable_parameters(model)
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}



    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        metric = (evaluate.load(data_args.metric_name))
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        metric = evaluate.load("accuracy")
        logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        metrics = trainer.predict(predict_dataset, metric_key_prefix="predict").metrics
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    shutil.rmtree(training_args.output_dir)







if __name__ == "__main__":
    main()
