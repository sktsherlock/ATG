import argparse
import torch
import sys
import os
import dgl
import numpy as np
import wandb
import os
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GraphData import Evaluator, split_edge, Logger, load_data
from LinkTask import linkprediction
from Utils.model_config import add_common_args


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def args_init():
    argparser = argparse.ArgumentParser(
        "Link-Prediction for SAGE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    argparser.add_argument('--hidden_channels', type=int, default=256)
    argparser.add_argument('--batch_size', type=int, default=2 * 1024)
    argparser.add_argument('--neg_len', type=str, default='5000')
    argparser.add_argument("--link_path", type=str, default="Data/LinkPrediction/Movies/", required=True,
                        help="Path to save the splitting for the link prediction tasks")
    return argparser



class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def main():
    argparser = args_init()
    args = argparser.parse_args()
    wandb.config = args
    wandb.init(config=args, reinit=True)
    print(args)

    if not os.path.exists(f'{args.link_path}{args.neg_len}/'):
        os.makedirs(f'{args.link_path}{args.neg_len}/')

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else 'cpu')

    # Load the graph
    if args.graph_path == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=args.graph_path)
        graph, _ = data[0]
    else:
        graph = dgl.load_graphs(f'{args.graph_path}')[0][0]


    edge_split = split_edge(graph, test_ratio=args.test_ratio, val_ratio=0.01, neg_len=args.neg_len, path=args.link_path)

    torch.manual_seed(42)
    idx = torch.randperm(edge_split['train']['source_node'].numel())[:len(edge_split['valid']['source_node'])]
    edge_split['eval_train'] = {
        'source_node': edge_split['train']['source_node'][idx],
        'target_node': edge_split['train']['target_node'][idx],
        'target_node_neg': edge_split['valid']['target_node_neg'],
    }

    train_edge_index = torch.stack((edge_split['train']['source_node'], edge_split['train']['target_node']), dim=1).t()

    feat = torch.from_numpy(np.load(args.feature).astype(np.float32)).to(device)

    adj_t = SparseTensor.from_edge_index(train_edge_index).t()
    adj_t = adj_t.to_symmetric().to(device)
    # Load the GCN model
    model = SAGE(feat.shape[1], args.n_hidden, args.n_hidden, args.n_layers, args.dropout).to(device)

    predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1,
                              3, args.dropout).to(device)

    evaluator = Evaluator()
    # logger = Logger(args.n_runs, args)
    loggers = {
        'Hits@1': Logger(args.n_runs, args),
        'Hits@3': Logger(args.n_runs, args),
        'Hits@10': Logger(args.n_runs, args),
        'MRR': Logger(args.n_runs, args),
    }

    for run in range(args.n_runs):
        model.reset_parameters()
        predictor.reset_parameters()

        loggers = linkprediction(args, adj_t, edge_split, model, predictor, feat, evaluator, loggers, run, args.neg_len)


        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)


    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)


if __name__ == "__main__":
    main()
