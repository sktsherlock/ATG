import argparse
import numpy as np
import pandas as pd
import torch
import os
import dgl
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, TrainingArguments, PreTrainedModel, Trainer, DataCollatorWithPadding, \
    AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset, load_dataset


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def infonce(anchor, sample, tau=0.2):
    sim = _similarity(anchor, sample) / tau
    num_nodes = anchor.shape[0]
    device = anchor.device
    pos_mask = torch.eye(num_nodes, dtype=torch.float32).to(device)
    neg_mask = 1. - pos_mask
    assert sim.size() == pos_mask.size()  # sanity check
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return -loss.mean()


class CLModel(PreTrainedModel):
    def __init__(self, PLM, dropout=0.0, cl_dim=128):
        super().__init__(PLM.config)
        self.dropout = nn.Dropout(dropout)
        hidden_dim = PLM.config.hidden_size
        self.text_encoder = PLM

        self.project = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cl_dim))

    def forward(self, input_ids, attention_mask, nb_input_ids, nb_attention_mask):
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        center_node_emb = self.dropout(center_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, output_hidden_states=True
        )

        toplogy_emb = self.dropout(toplogy_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        center_contrast_embeddings = self.project(center_node_emb)
        toplogy_contrast_embeddings = self.project(toplogy_emb)

        return center_contrast_embeddings, toplogy_contrast_embeddings


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # forward pass
        center_contrast_embeddings, toplogy_contrast_embeddings = model(**inputs)
        # compute
        loss = infonce(center_contrast_embeddings, toplogy_contrast_embeddings)
        return loss


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(
        description='Process text data and save the overall representation as an NPY file.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file', default="/dataintent/local/user/v-yinju/haoyan/Data/Movies/Movies.csv")
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
    parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny', required=True,
                        help='Name or path of the Huggingface model')
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
    parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to the NPY File')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
    parser.add_argument('--fp16', type=bool, default=True, help='if fp16')
    parser.add_argument('--cls', action='store_true', help='whether use first token to represent the whole text')
    parser.add_argument('--unfreeze_layers', type=int, default=2, help='Maximum length of the text for language models')
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    parser.add_argument("--graph_path", type=str, default="/dataintent/local/user/v-yinju/haoyan/Data/Movies/MoviesGraph.pt", help="The datasets to be implemented.")
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

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else 'cpu')

    graph = dgl.load_graphs(f'{args.graph_path}')[0][0].to(device)
    graph = dgl.to_bidirected(graph).to(device)

    neighbours = list(graph.adjacency_matrix_scipy().tolil().rows)


    class TopologyDataset(torch.utils.data.Dataset):
        def __init__(self, data, neighbours):
            super().__init__()
            self.dataset = data  # 存储传入的dataset
            self.neighbours = neighbours

        def __getitem__(self, node_id):
            item = self.dataset[node_id]
            neighbors = self.neighbours[node_id]
            k = np.random.choice(neighbors, 1)[0]
            item['nb_input_ids'] = self.dataset['input_ids'][k]
            item['nb_attention_mask'] = self.dataset['attention_mask'][k]
            return item

        def __len__(self):
            return len(self.dataset)  # 返回dataset的长度

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


    if args.pretrain_path is not None:
        model = AutoModel.from_pretrained(f'{args.pretrain_path}').to(device)
        print('Loading model from the path: {}'.format(args.pretrain_path))
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=access_token).to(device)


    train_data = TopologyDataset(dataset, neighbours).to(device)


    training_args = TrainingArguments(
        output_dir=cache_path,
        do_train=True,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        fp16=args.fp16,
        per_device_train_batch_size=args.batch_size
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()


    if args.save_path is not None:
        save_path = args.save_path + args.model_name_or_path.split('/')[-1].replace("-", "_") + '/' + f'lr_{training_args.learning_rate}_e_{training_args.num_train_epochs}_b_{training_args.per_device_train_batch_size}_u{args.unfreeze_layers}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")

        model.save_pretrained(save_path)



if __name__ == "__main__":
    main()
