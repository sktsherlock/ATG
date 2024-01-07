from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


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


class MLP(nn.Module):
    def __init__(
            self,
            in_feats,
            n_layers,
            n_hidden,
            activation,
            dropout=0.0,
            input_drop=0.0,
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
        self.input_drop = nn.Dropout(input_drop)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)

        return self.linears[-1](h), feat
