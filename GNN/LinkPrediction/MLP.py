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

from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphData import Evaluator, split_edge, Logger, load_data
from LinkTask import linkprediction
from Utils.model_config import add_common_args


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


def train(predictor, x, split_edge, optimizer, batch_size):
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(x.device)
    target_edge = split_edge['train']['target_node'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(x[src], x[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, x.size(0), src.size(), dtype=torch.long,
                                device=x.device)
        neg_out = predictor(x[src], x[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size, neg_len):
    predictor.eval()

    def test_split(split):
        source = split_edge[split]['source_node'].to(x.device)
        target = split_edge[split]['target_node'].to(x.device)
        target_neg = split_edge[split]['target_node_neg'].to(x.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(x[src], x[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(x[src], x[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({'y_pred_pos': pos_pred, 'y_pred_neg': neg_pred})

    train_results = test_split('eval_train', neg_len)
    valid_results = test_split('valid', neg_len)
    test_results = test_split('test', neg_len)

    Hits1 = train_results['hits@1_list'].mean().item(), valid_results['hits@1_list'].mean().item(), test_results[
        'hits@1_list'].mean().item()
    Hits5 = train_results['hits@5_list'].mean().item(), valid_results['hits@5_list'].mean().item(), test_results[
        'hits@5_list'].mean().item()
    Hits10 = train_results['hits@10_list'].mean().item(), valid_results['hits@10_list'].mean().item(), test_results[
        'hits@10_list'].mean().item()
    MRR = train_results['mrr_list'].mean().item(), valid_results['mrr_list'].mean().item(), test_results[
        'mrr_list'].mean().item()
    results = {
        'Hits@1': Hits1,
        'Hits@5': Hits5,
        'Hits@10': Hits10,
        'MRR': MRR
    }

    return results



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

    feat = torch.from_numpy(np.load(args.feature).astype(np.float32)).to(device)

    predictor = LinkPredictor(args.n_hidden, args.n_hidden, 1,
                              3, args.dropout).to(device)

    evaluator = Evaluator()
    # logger = Logger(args.n_runs, args)
    loggers = {
        'Hits@1': Logger(args.n_runs, args),
        'Hits@5': Logger(args.n_runs, args),
        'Hits@10': Logger(args.n_runs, args),
        'MRR': Logger(args.n_runs, args),
    }

    for run in range(args.n_runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.n_epochs):
            loss = train(predictor, feat, split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                result = test(predictor, feat, split_edge, evaluator,
                              args.batch_size)

                for key in loggers.keys():
                    loggers[key].add_result(run, result[key])

                if epoch % args.log_every == 0:
                    for key in loggers.keys():
                        train_result, valid_result, test_result = result[key]
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {train_result:.4f}, '
                              f'Valid: {valid_result:.4f}, '
                              f'Test: {test_result:.4f}')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)


    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
