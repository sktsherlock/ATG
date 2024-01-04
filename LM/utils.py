from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import torch


def split_dataset(nodes_num, train_ratio, val_ratio, data_name=None):
    if data_name == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=data_name)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        _, labels = data[0]
        labels = labels[:, 0]
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
        labels = None

    return train_idx, val_idx, test_idx, labels
