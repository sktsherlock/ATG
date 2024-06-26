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
from GNN.SIGIR.LPG import classification


class LPSAGE(nn.Module):
    def __init__(self,
                 in_LLM_feats,
                 in_PLM_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 ):
        super(LPSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        for i in range(n_layers):
            if i == 0:
                in_hidden = in_LLM_feats
            elif i == 1:
                in_hidden = in_PLM_feats
            else:
                in_hidden = n_hidden

            if i == n_layers - 1:
                out_hidden = n_classes
            elif i == 0:
                out_hidden = in_PLM_feats
            else:
                out_hidden = n_hidden

            self.convs.append(dglnn.SAGEConv(in_hidden, out_hidden, aggregator_type))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, graph, LLM_feat, PLM_feat):
        h = LLM_feat

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                if i == 0:
                    h = h + self.norms[i](PLM_feat)
                h = self.activation(h)
                h = self.dropout(h)

        return h


def args_init():
    argparser = argparse.ArgumentParser(
        "RevGAT Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--n-runs", type=int, default=3, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
    )
    argparser.add_argument(
        "--lr", type=float, default=0.005, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=256, help="number of hidden units"
    )
    argparser.add_argument(
        "--aggregator", type=str, default="mean", choices=["mean", "gcn", "pool", "lstm"],
        help="Specify the aggregator option of SAGE"
    )
    argparser.add_argument(
        "--no-attn-dst", type=bool, default=True, help="Don't use attn_dst."
    )
    argparser.add_argument(
        "--n-heads", type=int, default=3, help="number of heads"
    )
    argparser.add_argument(
        "--attn-drop", type=float, default=0.0, help="attention drop rate"
    )
    argparser.add_argument(
        "--edge-drop", type=float, default=0.0, help="edge drop rate"
    )
    argparser.add_argument(
        "--alpha", type=float, default=0.5, help="Tradeoff the LLM and the PLM"
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
        "--early_stop_patience", type=int, default=None,
        help="when to stop the  training loop to be aviod of the overfiting"
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
        "--PLM_feature", type=str, default=None, help="Use PLM embedding as feature"
    )
    argparser.add_argument(
        "--LLM_feature", type=str, default=None, help="Use LLM embedding as feature"
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

    PLM_feat = th.from_numpy(np.load(args.PLM_feature).astype(np.float32)).to(device)
    LLM_feat = th.from_numpy(np.load(args.LLM_feature).astype(np.float32)).to(device)

    n_classes = (labels.max() + 1).item()
    print(
        f"Number of classes {n_classes}, Number of PLM features {PLM_feat.shape[1]}, Number of LLM features {LLM_feat.shape[1]}")

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

    GNN = LPSAGE(LLM_feat.shape[1], PLM_feat.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu,
                 dropout=args.dropout, aggregator_type=args.aggregator).to(device)

    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in GNN.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")

    for run in range(args.n_runs):
        set_seed(args.seed)
        GNN.reset_parameters()
        val_result, test_result = classification(
            args, graph, GNN, PLM_feat, LLM_feat, labels, train_idx, val_idx, test_idx, run + 1
        )
        wandb.log({f'Val_{args.metric}': val_result, f'Test_{args.metric}': test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val accuracy: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test accuracy: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})


if __name__ == "__main__":
    main()
