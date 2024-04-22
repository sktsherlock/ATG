import sys
import copy
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import argparse
import wandb
import torch as th
import numpy as np
import torch.nn.functional as F
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphData import load_data, set_seed
from NodeClassification import classification
from Utils.model_config import add_common_args


# 模型定义模块
class GCN(nn.Module):
    def __init__(
            self,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            activation,
            dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(
                dglnn.GraphConv(in_hidden, out_hidden, "both")
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, graph, feat):
        h = feat

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


# 参数定义模块
def args_init():
    argparser = argparse.ArgumentParser(
        "GCN Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    return argparser


def main():
    argparser = args_init()
    args = argparser.parse_args()
    wandb.init(config=args, reinit=True)

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() else 'cpu')

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name, fewshots=args.fewshots)

    # add reverse edges, tranfer to the  undirected graph
    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    # 定义可观测图数据，用于inductive实验设置；
    observe_graph = copy.deepcopy(graph)


    if args.inductive:
        # 构造Inductive Learning 实验条件

        isolated_nodes = th.cat((val_idx, test_idx))
        sort_isolated_nodes, _ = th.sort(isolated_nodes)
        # 从图中删除指定节点
        observe_graph.remove_nodes(sort_isolated_nodes)

        # 添加相同数量的孤立节点
        observe_graph.add_nodes(len(sort_isolated_nodes))
        print(observe_graph)
        print('***************')
        print(graph)


    # add self-loop
    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device) if args.feature is not None else graph.ndata['feat'].to(device)
    n_classes = (labels.max()+1).item()
    print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)

    # run
    val_results = []
    test_results = []

    # Model implementation
    model = GCN(feat.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)
    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")
    # 确定所训练的模型保存的地址
    save_path = args.exp_path

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        exp_seed_path = os.path.join(save_path, f'Seed{args.seed}/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f'The save_path now is {save_path}')
        model.reset_parameters()
        val_result, test_result = classification(
            args, graph, observe_graph, model, feat, labels, train_idx, val_idx, test_idx, run+1, exp_seed_path
        )
        wandb.log({f'Val_{args.metric}': val_result, f'Test_{args.metric}': test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val {args.metric}: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test {args.metric}: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})


if __name__ == "__main__":
    main()
