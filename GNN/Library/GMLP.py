import torch.nn as nn
import argparse
import wandb
import torch as th
import numpy as np
import torch.nn.functional as F
import sys
import os
import torch.optim as optim
import time
import dgl.nn.pytorch as dglnn
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate, ncontrast
from GraphData import load_data, set_seed


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = th.eye(x_dis.shape[0]).cuda()
    x_sum = th.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = th.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


class MLP(nn.Module):
    def __init__(
            self,
            in_feats,
            n_layers,
            n_hidden,
            activation,
            dropout=0.0,
            input_drop=0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden

            self.linears.append(nn.Linear(in_hidden, out_hidden))

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, feat):
        h = self.input_drop(feat)

        for i in range(self.n_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)

        return self.linears[-1](h)


class GAdapter(nn.Module):
    def __init__(self, model, hidden, n_labels):
        super(GAdapter, self).__init__()
        self.mlp = model
        self.classifier = nn.Linear(hidden, n_labels)

    def reset_parameters(self):
        self.mlp.reset_parameters()

        self.classifier.reset_parameters()

    def forward(self, x):
        feature_cls = self.mlp(x)

        class_feature = self.classifier(feature_cls)

        return class_feature, feature_cls


def get_batch(feat, adj_label, batch_size, train_idx):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = th.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(th.long).cuda()
    rand_indx[0:len(train_idx)] = train_idx
    features_batch = feat[rand_indx]
    adj_label_batch = adj_label[rand_indx, :][:, rand_indx]
    return features_batch, adj_label_batch


def train(model, labels, train_idx, optimizer, args, feat, graph):
    features_batch, adj_label_batch = get_batch(feat, graph, batch_size=args.batch_size, train_idx=train_idx)

    model.train()
    optimizer.zero_grad()

    output, embeddings = model(features_batch)
    x_dis = get_feature_dis(embeddings)
    loss_train_class = cross_entropy(output[train_idx], labels[train_idx])
    loss_Ncontrast = ncontrast(x_dis, adj_label_batch, tau=args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.alpha

    loss_train.backward()
    optimizer.step()

    return loss_train_class, loss_Ncontrast, loss_train, output


@th.no_grad()
def evaluate(
        model, feat, labels, train_idx, val_idx, test_idx, metric='accuracy', label_smoothing=0.1, average=None
):
    model.eval()
    with th.no_grad():
        pred, _ = model(feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing)
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing)

    train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], metric, average=average)
    val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], metric, average=average)

    return train_results, val_results, test_results, val_loss, test_loss


def classification(
        args, model, feat, labels, train_idx, val_idx, test_idx, n_running, graph):
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

        loss_train_class, loss_Ncontrast, loss_train, output = train(model, labels, train_idx, optimizer, args, feat, graph)

        if epoch % args.eval_steps == 0:
            (
                train_result,
                val_result,
                test_result,
                val_loss,
                test_loss,
            ) = evaluate(
                model,
                feat,
                labels,
                train_idx,
                val_idx,
                test_idx,
                args.metric,
                args.label_smoothing,
                args.average
            )
            wandb.log(
                {'Loss_train_class': loss_train_class, 'Loss_Ncontrast': loss_Ncontrast,'Train_loss': loss_train,'Val_loss': val_loss, 'Test_loss': test_loss, 'Train_result': train_result,
                 'Val_result': val_result, 'Test_result': test_result})
            lr_scheduler.step(loss_train)

            toc = time.time()
            total_time += toc - tic

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_result = val_result
                final_test_result = test_result

            if args.early_stop_patience is not None:
                if stopper.step(val_loss):
                    break

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                    f"Loss: {loss_train.item():.4f}\n"
                    f"Train/Val/Test loss: {loss_train:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best Val/Final Test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
                )

    print("*" * 50)
    print(f"Best val acc: {best_val_result}, Final test acc: {final_test_result}")
    print("*" * 50)

    return best_val_result, final_test_result


def args_init():
    argparser = argparse.ArgumentParser(
        "Adapter Config",
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
        "--teacher-n-hidden", type=int, default=256, help="number of teacher models hidden units"
    )
    argparser.add_argument(
        "--teacher-layers", type=int, default=5, help="number of teacher models hidden units"
    )
    argparser.add_argument(
        "--teacher-n-heads", type=int, default=3, help="number of teacher models hidden units"
    )
    argparser.add_argument(
        "--teacher-lr", type=float, default=0.005, help="learning rate"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--input_drop", type=float, default=0.2, help="dropout rate"
    )
    argparser.add_argument(
        "--batch", type=int, default=64, help="number of hidden units"
    )
    argparser.add_argument(
        "--min-lr", type=float, default=0.0001, help="the min learning rate"
    )
    argparser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="the smoothing factor"
    )
    argparser.add_argument(
        "--alpha", type=float, default=2.0, help="To control the ratio of Ncontrast loss"
    )
    argparser.add_argument('--batch_size', type=int, default=2048,
                           help='batch size')
    argparser.add_argument('--tau', type=float, default=1.0,
                           help='temperature for Ncontrast loss')
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
        "--feature", type=str, default=None, help="Use LM embedding as feature", required=True
    )
    argparser.add_argument(
        "--label_embedding", type=str, default=None, help="Use label  embedding as label"
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
    argparser.add_argument(
        "--save_path", type=str, default='/dataintent/local/user/v-haoyan1/Model/Student/',
        help="Path to save the Student Model"
    )
    argparser.add_argument(
        "--save", type=bool, default=False, help="Whether to save the student model."
    )
    # ! Split dataset
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

    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    # add self-loop
    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device)

    n_classes = (labels.max() + 1).item()
    print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    in_features = feat.shape[1]

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
    set_seed(args.seed)
    Adapter = MLP(in_features, n_layers=args.n_layers, n_hidden=args.n_hidden, activation=F.relu,
                  dropout=args.dropout, input_drop=args.input_drop).to(device)

    Model = GAdapter(Adapter, hidden=args.n_hidden, n_labels=n_classes).to(device)

    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in Model.parameters() if p.requires_grad]
    )
    print(f"Number of the student model params: {TRAIN_NUMBERS}")

    val_results = []
    test_results = []

    for run in range(args.n_runs):
        Model.reset_parameters()
        val_result, test_result = classification(args, feat=feat, labels=labels, model=Model,n_running=run, test_idx=test_idx,
                                                 train_idx=train_idx, val_idx=val_idx, graph=graph)
        wandb.log({f'Val_{args.metric}': val_result, f'Test_{args.metric}': test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val accuracy: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test accuracy: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})


if __name__ == "__main__":
    main()
