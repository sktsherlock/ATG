import argparse
import numpy as np
import pandas as pd
import torch
import os
import dgl
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace as SN
from torch_sparse import SparseTensor
import shutil
from ogb.nodeproppred import DglNodePropPredDataset
from Task import TopologicalCL
from transformers import AutoTokenizer, AutoModel, TrainingArguments, PreTrainedModel, Trainer, DataCollatorWithPadding, \
    AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset, load_dataset


class MeanEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.encoder = model


    def mean_pooling(self, token_embeddings, attention_mask):
        # Mask out padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        masked_token_embeddings = token_embeddings * input_mask_expanded
        # Calculate mean pooling
        mean_embeddings = masked_token_embeddings.sum(dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return mean_embeddings


    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        node_mean_emb = (self.mean_pooling(outputs.last_hidden_state, attention_mask))
        return TokenClassifierOutput(logits=node_mean_emb)


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

    def forward(self, input_ids=None, attention_mask=None, nb_input_ids=None, nb_attention_mask=None): # , nb_input_ids, nb_attention_mask
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        center_node_emb = self.dropout(torch.mean(center_node_outputs.last_hidden_state, dim=1))

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, output_hidden_states=True
        )

        toplogy_emb = self.dropout(torch.mean(toplogy_node_outputs.last_hidden_state, dim=1))

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


class Sequence:
    def __init__(self, cf):
        self.ndata = {}
        self.n_nodes = cf['n_nodes']
        self.max_length = cf['max_length']
        self.inductive = False
        self.graph_path = cf['graph_path']
        self.token_folder = cf['token_folder']


        self.info = {
            'input_ids': SN(shape=(self.n_nodes, self.max_length), type=np.uint16),
            'attention_mask': SN(shape=(self.n_nodes, self.max_length), type=bool),
        }
        for k, info in self.info.items():
            info.path = f'{self.token_folder}{k}.npy'


    def init(self):
        self._load_data_fields()
        self.neighbours = self.get_neighbours()
        if self.inductive:
            self.edge_index = self.get_train_edge()
        return self


    def get_neighbours(self):
        g = dgl.load_graphs(self.graph_path)[0][0]
        g = dgl.to_bidirected(g)

        neighbours = list(g.adjacency_matrix_scipy().tolil().rows)
        return neighbours



    def _load_data_fields(self):
        for k in self.info:
            i = self.info[k]
            try:
                self.ndata[k] = np.load(i.path, allow_pickle=True)  # np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
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
        item['nb_attention_mask'] = _load('attention_mask')
        item['nb_input_ids'] = torch.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        return item

    def __getitem__(self, k):
        return self.ndata[k]


class TopologyDataset(torch.utils.data.Dataset):
    def __init__(self, data: Sequence): #neighbours
        super().__init__()
        self.d = data  # 存储传入的dataset
        # self.neighbours = neighbours

    def __getitem__(self, node_id):
        item = self.d.get_tokens(node_id)
        neighbours = self.d.neighbours[node_id]
        if neighbours:
            k = np.random.choice(neighbours, 1)[0]
            item = self.d.get_nb_tokens(item, k)
        else:
            # Do the self contrastive learning
            k = node_id
            item = self.d.get_nb_tokens(item, k)
        return item

    def __len__(self):
        return self.d.n_nodes


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='TCL and Tuning for the language models.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file', default="Data/Movies/Movies.csv", required=True)
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
    parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny', required=True,
                        help='Name or path of the Huggingface model')
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file', required=True)
    parser.add_argument('--path', type=str, default='./', help='Path to the NPY File', required=True)
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to the NPY File')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the NPY File')
    parser.add_argument('--token_folder', type=str, default='/dataintent/local/user/v-yinju/haoyan/Token/Movies/', help='Path to the NPY File', required=True)
    parser.add_argument('--save_path', type=str, default=None, help='Path to the NPY File')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length of the text for language models')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of batch size for inference')
    parser.add_argument('--fp16', type=bool, default=True, help='if fp16')
    parser.add_argument('--cls', action='store_true', help='whether use first token to represent the whole text')
    parser.add_argument('--unfreeze_layers', type=int, default=2, help='Maximum length of the text for language models')
    parser.add_argument("--drop_out", type=float, default=0.5, help="Dropout  Ratio.")
    parser.add_argument("--learning_rate", type=float, default=2e-05, help="learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of pretrain epochs to train')

    parser.add_argument("--graph_path", type=str, default="Data/Movies/MoviesGraph.pt", help="The datasets to be implemented.", required=True)
    # 解析命令行参数
    args = parser.parse_args()
    csv_file = args.csv_file
    text_column = args.text_column
    model_name = args.model_name
    name = args.name
    max_length = args.max_length
    access_token = "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"
    tokenizer_name = args.tokenizer_name
    output_dir = args.output_dir + args.model_name.split('/')[-1].replace("-", "_") + '/' + f'd_{args.drop_out}_w_{args.warmup_ratio}_lr_{args.learning_rate}_e_{args.num_train_epochs}_b_{args.batch_size}_u{args.unfreeze_layers}'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))
    pretrained_model_path = os.path.join(base_dir, args.path)
    token_folder = os.path.join(base_dir, args.token_folder)
    if not os.path.exists(pretrained_model_path):
        os.makedirs(pretrained_model_path)
    if not os.path.exists(token_folder):
        os.makedirs(token_folder)
    data_files = os.path.join(base_dir, csv_file)

    # 读取CSV文件
    df = pd.read_csv(data_files)
    text_data = df[text_column].tolist()

    # 加载模型和分词器
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=access_token,  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token,  trust_remote_code=True)

    # 编码文本数据并转为数据集
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(text_data, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    dataset = Dataset.from_dict(tokenized)
    print(dataset)

    for k in tokenized.data:
        with open(os.path.join(token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized.data[k])

    # 将本地token file读入到数据集中
    cf = {'n_nodes': len(df), 'max_length': args.max_length, 'graph_path': args.graph_path, 'token_folder': token_folder}

    d = Sequence(cf).init()
    train_data = TopologyDataset(d)


    if args.pretrain_path is not None:
        encoder = AutoModel.from_pretrained(f'{args.pretrain_path}')
        print('Loading model from the path: {}'.format(args.pretrain_path))
    else:
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=access_token)


    if args.unfreeze_layers is not None:
        for param in encoder.parameters():
            param.requires_grad = False

        trainable_params = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad
            )
        assert trainable_params == 0
        for param in encoder.encoder.layer[-args.unfreeze_layers:].parameters():
            param.requires_grad = True

        trainable_params = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad
        )
        print(f" Pass the freeze layer, the LM Encoder  parameters are {trainable_params}")

    model = TopologicalCL(
        encoder,
        dropout=args.drop_out,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        fp16=args.fp16,
        save_total_limit=None,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=10,
        warmup_ratio=0.1,
        per_device_eval_batch_size=args.batch_size * 10,
        learning_rate=args.learning_rate,
        save_strategy='epoch',
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()

    if pretrained_model_path is not None:
        print(f'Save the model under TCL in {pretrained_model_path}')
        encoder.save_pretrained(pretrained_model_path)
    else:
        encoder.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
