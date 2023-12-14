import numpy as np
import torch as th
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

def split_graph(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


def load_data(graph_path, train_ratio=0.6, val_ratio=0.2, name=None):
    if name == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=name)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        labels = labels[:, 0]
    else:
        # load the graph from local path
        graph = dgl.load_graphs(graph_path)[0][0]
        labels = graph.ndata['label']
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), train_ratio, val_ratio)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    return graph, labels, train_idx, val_idx, test_idx
