import sys
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import argparse
import wandb
import torch as th
import numpy as np
import torch.nn.functional as F
import os
import time
import torch.optim as optim


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate
from GraphData import load_data


def compute_fusion_accuracy_overlap(preds1, preds2, labels):
    total_samples = len(labels)
    overlap_count = 0

    for i in range(total_samples):
        if preds1[i] == labels[i] or preds2[i] == labels[i]:
            overlap_count += 1


    overlap_rate = overlap_count / total_samples

    return overlap_rate

def compute_accuracy_overlap(preds1, preds2, labels):
    total_samples = len(labels)
    overlap_count = 0

    for i in range(total_samples):
        if preds1[i] == labels[i] and preds2[i] == labels[i]:
            overlap_count += 1


    overlap_rate = overlap_count / total_samples

    return overlap_rate


def compute_overlap_rate(preds1, preds2):
    total_samples = len(preds1)
    overlap_count = 0

    for i in range(total_samples):
        if preds1[i] == preds2[i]:
            overlap_count += 1

    overlap_rate = overlap_count / total_samples
    return overlap_rate

def train(model, graph, feat, labels, train_idx, optimizer, label_smoothing):
    model.train()

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss, pred



@th.no_grad()
def evaluate(
    model, graph, feat, labels, train_idx, val_idx, test_idx, metric='accuracy', label_smoothing=0.1, average=None
):
    model.eval()
    with th.no_grad():
        pred = model(graph, feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing)
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing)

    train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], metric, average=average)
    val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], metric, average=average)

    return train_results, val_results, test_results, val_loss, test_loss, pred





def classification(
        args, graph, model, feat, labels, train_idx, val_idx, test_idx):
    if args.early_stop_patience is not None:
        stopper = EarlyStopping(patience=args.early_stop_patience)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        verbose=True,
        min_lr=args.min_lr,
    )

    # training loop
    total_time = 0
    best_val_result, final_test_result, best_val_loss = 0, 0, float("inf")


    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        if args.warmup_epochs is not None:
            adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)

        train_loss, pred = train(
            model, graph, feat, labels, train_idx, optimizer, label_smoothing=args.label_smoothing
        )
        if epoch % args.eval_steps == 0:
            (
                train_result,
                val_result,
                test_result,
                val_loss,
                test_loss,
                prediction,
            ) = evaluate(
                model,
                graph,
                feat,
                labels,
                train_idx,
                val_idx,
                test_idx,
                args.metric,
                args.label_smoothing,
                args.average
            )
            wandb.log({'Train_loss': train_loss, 'Val_loss': val_loss, 'Test_loss': test_loss, 'Train_result': train_result, 'Val_result': val_result, 'Test_result': test_result})
            lr_scheduler.step(train_loss)

            toc = time.time()
            total_time += toc - tic

            if val_result > best_val_result:
                best_val_result = val_result
                final_test_result = test_result
                best_prediction = prediction


            if args.early_stop_patience is not None:
                if stopper.step(val_result):
                    break

            if epoch % args.log_every == 0:
                print(
                    f"Loss: {train_loss.item():.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best Val/Final Test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
                )

    print("*" * 50)
    print(f"Best val acc: {best_val_result}, Final test acc: {final_test_result}")
    print("*" * 50)

    return best_val_result, final_test_result, best_prediction


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 ):
        super(GraphSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

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


def args_init():
    argparser = argparse.ArgumentParser(
        "GrahSAGE Config",
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
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--aggregator", type=str, default="mean", choices=["mean", "gcn", "pool", "lstm"], help="Specify the aggregator option"
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

    feat1 = th.from_numpy(np.load(args.PLM_feature).astype(np.float32)).to(device)
    feat2 = th.from_numpy(np.load(args.LLM_feature).astype(np.float32)).to(device)


    n_classes = (labels.max()+1).item()
    print(f"Number of classes {n_classes}, Number of feat1 {feat1.shape[1]}, Number of feat2 {feat2.shape[1]}")

    graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    labels = labels.to(device)
    graph = graph.to(device)


    # Model implementation
    model = GraphSAGE(feat1.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, aggregator_type=args.aggregator).to(device)
    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")


    model.reset_parameters()
    val_result, test_result, prediction1 = classification(
        args, graph, model, feat1, labels, train_idx, val_idx, test_idx
    )
    wandb.log({f'Feat1_Val_{args.metric}': val_result, f'Feat1_Test_{args.metric}': test_result})



    model = GraphSAGE(feat2.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, aggregator_type=args.aggregator).to(device)
    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")

    model.reset_parameters()
    val_result, test_result, prediction2 = classification(
        args, graph, model, feat2, labels, train_idx, val_idx, test_idx
    )
    wandb.log({f'Feat2_Val_{args.metric}': val_result, f'Feat2_Test_{args.metric}': test_result})

    acc_overlap = compute_accuracy_overlap(th.argmax(prediction1[test_idx], dim=1), th.argmax(prediction2[test_idx], dim=1), labels[test_idx])
    print("Overlap Rate: {:.2f}".format(acc_overlap))
    wandb.log({f'Acc_overlap': acc_overlap})
    overlap = compute_overlap_rate(th.argmax(prediction1[test_idx], dim=1), th.argmax(prediction2[test_idx], dim=1))
    wandb.log({f'Overlap': overlap})
    acc_fusion = compute_fusion_accuracy_overlap(th.argmax(prediction1[test_idx], dim=1), th.argmax(prediction2[test_idx], dim=1), labels[test_idx])
    print("Fusion Acc Rate: {:.2f}".format(acc_fusion))
    wandb.log({f'Acc_fusion': acc_fusion})



if __name__ == "__main__":
    main()
