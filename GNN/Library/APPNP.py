"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import sys
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
from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(
                nn.Linear(in_hidden, out_hidden)
            )
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

    def forward(self, graph, features):
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.convs[0](h))

        for layer in self.convs[1:-1]:
            h = self.activation(layer(h))
        h = self.convs[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(graph, h)
        return h


# 参数定义模块
def args_init():
    argparser = argparse.ArgumentParser(
        "APPNP Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--edge_drop", type=float, default=0.5, help="edge propagation dropout"
    )
    argparser.add_argument(
        "--k", type=int, default=10, help="Number of propagation steps"
    )
    argparser.add_argument(
        "--alpha", type=float, default=0.1, help="Teleport Probability"
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--n-runs", type=int, default=3, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
    )
    argparser.add_argument(
        "--lr", type=float, default=1e-2, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=64, help="number of hidden units"
    )
    argparser.add_argument(
        "--input_dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--weight", type=bool, default=True, help="if False, no W."
    )
    argparser.add_argument(
        "--bias", type=bool, default=True, help="if False, no last layer bias."
    )
    argparser.add_argument(
        "--min-lr", type=float, default=0.0001, help="the min learning rate"
    )
    argparser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="the smoothing factor"
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    # ! default
    argparser.add_argument(
        "--log-every", type=int, default=20, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--eval_steps", type=int, default=1, help="eval in every epochs"
    )
    argparser.add_argument(
        "--early_stop_patience", type=int, default=None, help="when to stop the  training loop to be aviod of the overfiting"
    )
    argparser.add_argument(
        "--warmup_epochs", type=int, default=None, help="The warmup epochs"
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="The seed for the teacher models"
    )
    # ! Data related
    argparser.add_argument(
        "--data_name", type=str, default=None, help="The dataset name.",
    )
    argparser.add_argument(
        "--feature", type=str, default=None, help="Use LM embedding as feature",
    )
    argparser.add_argument(
        "--graph_path", type=str, default=None, help="The datasets to be implemented."
    )
    argparser.add_argument(
        "--undirected", type=bool, default=True, help="Whether to undirect the graph."
    )
    argparser.add_argument(
        "--selfloop", type=bool, default=True, help="Whether to add self loop in the graph."
    )
    argparser.add_argument(
        "--metric", type=str, default='accuracy', choices=['accuracy', 'precision', 'recall', 'f1'],
        help="The metric to be used."
    )
    argparser.add_argument(
        "--average", type=str, default='weighted', choices=['weighted', 'micro', 'macro', None]
    )
    # ! Split datasets
    argparser.add_argument(
        "--inductive", type=bool, default=False, help="Whether to do inductive learning experiments."
    )
    argparser.add_argument(
        "--train_ratio", type=float, default=0.6, help="training ratio"
    )
    argparser.add_argument(
        "--val_ratio", type=float, default=0.2, help="training ratio"
    )
    return argparser


def main():
    argparser = args_init()
    args = argparser.parse_args()
    wandb.init(config=args, reinit=True)

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() else 'cpu')

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name)


    if args.inductive:
        # 构造Inductive Learning 实验条件
        isolated_nodes = th.cat((val_idx, test_idx))
        sort_isolated_nodes, _ = th.sort(isolated_nodes)
        # 从图中删除指定节点
        graph.remove_nodes(sort_isolated_nodes)

        # 添加相同数量的孤立节点
        graph.add_nodes(len(sort_isolated_nodes))

    # add reverse edges, tranfer to the  undirected graph
    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    # add self-loop
    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device) if args.feature is not None else graph.ndata['feat'].to(device)
    n_classes = (labels.max()+1).item()
    print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_results = []
    test_results = []

    # Model implementation
    model = APPNP(feat.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu, args.input_dropout, args.edge_drop, args.alpha, args.k).to(device)
    print(model)
    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")

    for run in range(args.n_runs):
        set_seed(args.seed)
        model.reset_parameters()
        val_result, test_result = classification(
            args, graph, model, feat, labels, train_idx, val_idx, test_idx, run+1
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