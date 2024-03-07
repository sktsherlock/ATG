import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset
from GraphData import Evaluator, split_edge, Logger
import dgl
import numpy as np
import wandb
import os
from LinkPrediction import SAGE, GCN, LinkPredictor


class LPGNN(torch.nn.Module):
    def __init__(self, model, LLM_in_feats, PLM_in_feats, alpha=0.5, conv='SAGE'):
        super().__init__()
        self.GNN = model
        self.conv = conv
        if conv == 'SAGE':
            self.decomposition = SAGEConv(LLM_in_feats, PLM_in_feats, 'mean')
        else:
            self.decomposition = torch.nn.Linear(LLM_in_feats, PLM_in_feats)
        self.alpha = alpha

    def reset_parameters(self):
        self.GNN.reset_parameters()

        self.decomposition.reset_parameters()

    def forward(self, PLM_feat, LLM_feat, adj_t):  #
        # Decomposition the LLM features
        LLM_feat = self.decomposition(LLM_feat, adj_t) if self.conv == 'SAGE' else self.decomposition(LLM_feat)

        # Trade off the LLM_feat and the PLM_feat

        feat = self.alpha * LLM_feat + (1 - self.alpha) * PLM_feat
        # Extract outputs from the model
        h = self.GNN(feat, adj_t)

        return h


def train(model, predictor, PLM_feat, LLM_feat, adj_t, edge_split, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = edge_split['train']['edge'].to(PLM_feat.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(PLM_feat, LLM_feat, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, PLM_feat.size(0), edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, PLM_feat, LLM_feat, adj_t, edge_split, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(PLM_feat, LLM_feat, adj_t)

    pos_train_edge = edge_split['train']['edge'].to(h.device)
    pos_valid_edge = edge_split['valid']['edge'].to(h.device)
    neg_valid_edge = edge_split['valid']['edge_neg'].to(h.device)
    pos_test_edge = edge_split['test']['edge'].to(h.device)
    neg_test_edge = edge_split['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(PLM_feat, LLM_feat, adj_t)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='Link-Prediction for GNN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gnn_model', type=str, help='GNN MOdel', default='GCN')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--test_ratio', type=float, default=0.08)
    parser.add_argument('--val_ratio', type=float, default=0.02)
    parser.add_argument('--neg_len', type=str, default='2000')
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Tradeoff the LLM and the PLM"
    )
    parser.add_argument(
        "--conv_type", type=str, default='SAGE', help="Use PLM embedding as feature"
    )
    parser.add_argument("--PLM_feature", type=str, default=None, help="Use LM embedding as feature",)
    parser.add_argument("--LLM_feature", type=str, default=None, help="Use LLM embedding as feature")
    parser.add_argument("--path", type=str, default="/dataintent/local/user/v-yinju/haoyan/LinkPrediction/Movies/",
                        help="Path to save splitting")
    parser.add_argument("--graph_path", type=str, default="/dataintent/local/user/v-yinju/haoyan/Data/Movies/MoviesGraph.pt",
                        help="Path to load the graph")
    args = parser.parse_args()
    wandb.config = args
    wandb.init(config=args, reinit=True)
    print(args)


    if not os.path.exists(f'{args.path}{args.neg_len}/'):
        os.makedirs(f'{args.path}{args.neg_len}/')

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.graph_path == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=args.graph_path)
        graph, _ = data[0]
    else:
        graph = dgl.load_graphs(f'{args.graph_path}')[0][0]


    edge_split = split_edge(graph, test_ratio=0.08, val_ratio=0.02, path=args.path, neg_len=args.neg_len)

    PLM_feat = torch.from_numpy(np.load(args.PLM_feature).astype(np.float32)).to(device)
    LLM_feat = torch.from_numpy(np.load(args.LLM_feature).astype(np.float32)).to(device)


    edge_index = edge_split['train']['edge'].t()
    adj_t = SparseTensor.from_edge_index(edge_index).t()
    adj_t = adj_t.to_symmetric().to(device)


    if args.gnn_model == 'SAGE':
        model = SAGE(PLM_feat.size(1), args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.gnn_model == 'GCN':
        model = GCN(PLM_feat.size(1), args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    else:
        raise ValueError('Not implemented')
    model = LPGNN(model, LLM_feat.shape[1], PLM_feat.shape[1], args.alpha, args.conv_type).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='Movies')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, PLM_feat, LLM_feat, adj_t, edge_split, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, PLM_feat, LLM_feat, adj_t, edge_split, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)


if __name__ == "__main__":
    main()