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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GraphData import Evaluator, split_edge, Logger, load_data
from LinkPrediction import linkprediction
from Utils.model_config import add_common_args
from Library.GCN import GCN


def args_init():
    argparser = argparse.ArgumentParser(
        "Link-Prediction for GCN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    argparser.add_argument('--hidden_channels', type=int, default=256)
    argparser.add_argument('--batch_size', type=int, default=2* 1024)
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

    if not os.path.exists(f'{args.path}{args.neg_len}/'):
        os.makedirs(f'{args.path}{args.neg_len}/')

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load the graph
    if args.graph_path == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=args.graph_path)
        graph, _ = data[0]
    else:
        graph = dgl.load_graphs(f'{args.graph_path}')[0][0]


    edge_split = split_edge(graph, test_ratio=args.test_ratio, val_ratio=0.02, path=args.path, neg_len=args.neg_len)

    feat = torch.from_numpy(np.load(args.feature).astype(np.float32)).to(device)

    edge_index = edge_split['train']['edge'].t()
    adj_t = SparseTensor.from_edge_index(edge_index).t()
    print('The first adj_ t is: {}'.format(adj_t))
    adj_t = adj_t.to_symmetric().to(device)
    print('The second adj_t is:{}'.format(adj_t))

    model = GCN(feat.shape[1], args.n_hidden, args.n_hidden, args.n_layers, F.relu, args.dropout).to(device)

    predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1,
                              3, args.dropout).to(device)

    evaluator = Evaluator(name='Movies')
    loggers = {
        'Hits@10': Logger(args.n_runs, args),
        'Hits@50': Logger(args.n_runs, args),
        'Hits@100': Logger(args.n_runs, args),
    }

    for run in range(args.n_runs):
        model.reset_parameters()
        predictor.reset_parameters()

        loggers = linkpredition(args, adj_t, edge_split, model, predictor, feat, evaluator, loggers, run)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)


if __name__ == "__main__":
    main()
