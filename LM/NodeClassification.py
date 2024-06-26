import logging
import os
import random
import sys
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import evaluate
import numpy as np
os.environ['HF_DATASETS_OFFLINE'] = '1'
from datasets import load_dataset
import torch
from datasets import DatasetDict
from utils import split_dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    AutoConfig,
    HfArgumentParser,
    set_seed,
)

from Task import CLSClassifier, MEANClassifier, AdapterClassifier, GAdapterClassifier
from transformers.utils import send_example_telemetry

# check_min_version("4.35.0")
#
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

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
    prefix: Optional[str] = field(
        default=None, metadata={"help": "The dataname to be used for splitting dataaset. like ogbn-arxiv"}
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


def get_label_list(raw_dataset, split="train") -> List[int]:
    """Get the list of labels from a multi-label dataset"""

    label_column = raw_dataset[split]["label"]

    if isinstance(label_column[0], list):
        # For multi-label case
        label_list = [label for sample in label_column for label in sample]
        label_list = list(set(label_list))
    elif isinstance(label_column[0], int):
        # For single-label case with int values
        label_list = list(set(label_column))
    else:
        raise ValueError("Unsupported label type. Expected list or int.")

    return label_list


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
                                                                                                               "_") + '/' + f't_{data_args.train_ratio}_v_{data_args.val_ratio}_d_{model_args.drop_out}_w_{training_args.warmup_ratio}_lr_{training_args.learning_rate}_e_{training_args.num_train_epochs}_b_{training_args.per_device_train_batch_size}_u{model_args.unfreeze_layers}'
    training_args.evaluation_strategy = 'epoch'
    training_args.save_strategy = 'epoch'
    training_args.load_best_model_at_end = True
    training_args.save_total_limit = None
    training_args.do_train = True
    training_args.do_eval = True
    training_args.do_predict = True

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

    train_ids, val_ids, test_ids = split_dataset(nodes_num, data_args.train_ratio, data_args.val_ratio,
                                                 data_name=data_args.data_name)
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
    print(label_list)
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )



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
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
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
        model = MEANClassifier(
            encoder, num_labels,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean')
        )
    elif model_args.training_objective == 'GAdapter':
        Gadapter = torch.load(model_args.filename)
        Gadapter.input_drop = torch.nn.Dropout(0.0)

        model = GAdapterClassifier(
            encoder, adapter=Gadapter, n_labels=num_labels,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean'),
            resduial=model_args.resduial
        )
    elif model_args.training_objective == 'Adapter':
        adapter = torch.load(model_args.filename)
        model = AdapterClassifier(
            encoder, adapter=adapter,
            dropout=model_args.drop_out,
            loss_func=torch.nn.CrossEntropyLoss(label_smoothing=model_args.label_smoothing, reduction='mean')
        )
    else:
        raise ValueError("Training objective should be either CLS or Mean.")
    print('Model OK!')
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

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
        result["label"] = examples["label"]
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
            print("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    print("Shuffling the training dataset Finished")
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                print("Validation dataset not found. Falling back to test dataset for validation.")
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
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        print(f"Metric name: {data_args.metric_name}")
        metric = (evaluate.load('metrics/' + data_args.metric_name))
        print(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        metric = evaluate.load("metrics/accuracy")
        print("Using accuracy as classification score, you can use --metric_name to overwrite.")

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
        predictions = trainer.predict(predict_dataset)
        predicted_labels = predictions.predictions.argmax(axis=-1)  # 获取预测的标签

        # 创建 DataFrame 存储预测结果
        import pandas as pd
        df = pd.DataFrame({"True Label": predict_dataset['label'], "Predicted Label": predicted_labels})

        # 保存预测结果到 CSV 文件
        df.to_csv(f"{data_args.prefix}_predictions.csv", index=False)

        metrics = trainer.evaluate(eval_dataset=predict_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    if model_args.save_path is not None:
        save_path = model_args.save_path + model_args.model_name_or_path.split('/')[-1].replace("-",
                                                                                                "_") + '/' + f't_{data_args.train_ratio}_v_{data_args.val_ratio}_d_{model_args.drop_out}_w_{training_args.warmup_ratio}_lr_{training_args.learning_rate}_e_{training_args.num_train_epochs}_b_{training_args.per_device_train_batch_size}_u{model_args.unfreeze_layers}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Created directory: {save_path}")

        encoder.save_pretrained(save_path)
        logger.info("*** PLM Saved successfully ***")

    shutil.rmtree(training_args.output_dir)


if __name__ == "__main__":
    main()
