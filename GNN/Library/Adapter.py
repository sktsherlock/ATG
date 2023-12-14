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
from LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate, _contrastive_loss
from GraphData import load_data


def train(model, feat, labels, train_idx, optimizer, label_smoothing):
    model.train()

    optimizer.zero_grad()
    pred = model(feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss, pred


def training(
        args, student_model, teacher_model, graph, feat, label_embedding, train_idx, val_idx, test_idx, filename):
    optimizer = optim.AdamW(
        student_model.parameters(), lr=args.lr, weight_decay=args.wd
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
    best_val_loss = float("inf")
    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        # Training Loop
        student_model.train()

        if args.warmup_epochs is not None:
            adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)

        # teacher model
        with th.no_grad():  # 教师网络不用反向传播
            teacher_graph_preds = teacher_model(graph, feat)

        # student model forward
        student_preds = student_model(feat)
        student_graph_preds = student_model.graph_forward(feat)
        student_loss = cross_entropy((student_preds[train_idx], label_embedding[train_idx]))

        ditillation_loss = _contrastive_loss(student_graph_preds, teacher_graph_preds)

        loss = args.alpha * student_loss + (1 - args.alpha) * ditillation_loss

        optimizer.zero_grad()
        loss.backward()				#反向传播
        optimizer.step()			#参数优化

        student_model.eval()
        with th.no_grad():
            pred = student_model(feat)
        val_loss = cross_entropy(pred[val_idx], label_embedding[val_idx])
        test_loss = cross_entropy(pred[test_idx], label_embedding[test_idx])

        wandb.log({'Train_loss': loss, 'Test_loss': test_loss, 'Val_loss': val_loss, 'Distillation_loss': ditillation_loss})
        lr_scheduler.step(loss)
        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss

            th.save(student_model.state_dict(), filename)


    return best_val_loss, best_test_loss


class GraphAdapter(nn.Module):
    def __init__(
            self,
            in_feats,
            n_layers,
            n_hidden,
            activation,
            dropout=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else in_feats

            self.linears.append(nn.Linear(in_hidden, out_hidden))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def graph_forward(self, feat):
        h = feat

        for i in range(self.n_layers - 2):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)

        return h

    def forward(self, feat):
        h = feat

        for i in range(self.n_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)

        return h


class GCNTeacher(nn.Module):
    def __init__(
            self,
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            weight=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden

            self.convs.append(
                dglnn.GraphConv(in_hidden, out_hidden, "both", weight=weight, bias=True)
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
        "--dropout", type=float, default=0.5, help="dropout rate"
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
    # ! Data related
    argparser.add_argument(
        "--feature", type=str, default=None, help="Use LM embedding as feature", required=True
    )
    argparser.add_argument(
        "--label_embedding", type=str, default=None, help="Use label  embedding as label", required=True
    )
    argparser.add_argument(
        "--graph_path", type=str, default=None, help="The datasets to be implemented.", required=True
    )
    argparser.add_argument(
        "--metric", type=str, default='accuracy', choices=['accuracy', 'precision', 'recall', 'f1'],
        help="The metric to be used."
    )
    argparser.add_argument(
        "--average", type=str, default='weighted', choices=['weighted', 'micro', 'macro', None]
    )
    argparser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the Student Model", required=True
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
    graph, _, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio)

    feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device)
    label_embedding = th.from_numpy(np.load(args.label_embedding).astype(np.float32)).to(device)


    in_features = feat.shape[1]

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')


    # run
    val_results = []
    test_results = []

    # Model implementation
    student_model = GraphAdapter(in_features, n_layers=args.n_layers, n_hidden=args.n_hidden, activation=F.relu, dropout=args.dropout).to(device)

    teacher_model = GCNTeacher(in_features, args.n_hidden, args.n_layers-1, F.relu, dropout=args.dropout)

    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in student_model.parameters() if p.requires_grad]
    )
    print(f"Number of the student model params: {TRAIN_NUMBERS}")


    student_model.reset_parameters()
    teacher_model.reset_parameters()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(args.save_path, f"best_student_model_{timestamp}.pt")
    training(args, student_model, teacher_model, graph, feat, label_embedding, train_idx, val_idx, test_idx, filename)
    # Distil the Graph Knowledge to the Adapter






if __name__ == "__main__":
    main()
