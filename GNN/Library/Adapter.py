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
from RevGAT.model import RevGAT

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate, _contrastive_loss
from GraphData import load_data, set_seed


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

    return train_results, val_results, test_results, val_loss, test_loss


def teacher_training(args, teacher_model, graph, feat, label, train_idx, val_idx, test_idx, model_path):
    if os.path.exists(model_path):
        teacher_model.load_state_dict(th.load(model_path))
    else:
        # Start Training and save the trained model
        if args.early_stop_patience is not None:
            stopper = EarlyStopping(patience=50)
        optimizer = optim.AdamW(
            teacher_model.parameters(), lr=args.lr, weight_decay=0
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=100,
            verbose=True,
            min_lr=0.0001,
        )

        # training loop
        total_time = 0
        best_val_result, final_test_result, best_val_loss = 0, 0, float("inf")

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()

            if args.warmup_epochs is not None:
                adjust_learning_rate(optimizer, args.lr, epoch, 50)

            train_loss, pred = train(
                teacher_model, graph, feat, label, train_idx, optimizer, label_smoothing=0.1
            )
            train_result, val_result, test_result, val_loss, test_loss = evaluate(teacher_model, graph, feat, label,
                                                                                  train_idx, val_idx, test_idx,
                                                                                  metric=args.metric,
                                                                                  label_smoothing=0.1,
                                                                                  average=args.average)
            wandb.log({'Teacher_Train_loss': train_loss, 'Teacher_Val_loss': val_loss, 'Teacher_Test_loss': test_loss,
                       'Teacher_Train_result': train_result,
                       'Teacher_Val_result': val_result, 'Teacher_Test_result': test_result})
            lr_scheduler.step(train_loss)

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
                    f"Teacher Runing Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                    f"Loss: {train_loss.item():.4f}\n"
                    f"Teacher Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Teacher Train/Val/Test/Best Val/Final Test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
                )

        print("*" * 50)
        print('Teacher model training over.')
        print(f"Best val acc: {best_val_result}, Final test acc: {final_test_result}")
        print("*" * 50)

        th.save(teacher_model.state_dict(), model_path)

    return teacher_model


def student_training(
        args, student_model, teacher_model, graph, feat, labels, train_idx, val_idx, test_idx, filename):
    student_optimizer = optim.AdamW(
        student_model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        student_optimizer,
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

        # Training Loop
        student_model.train()

        if args.warmup_epochs is not None:
            adjust_learning_rate(student_optimizer, args.lr, epoch, args.warmup_epochs)

        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        # teacher model
        with th.no_grad():  # 教师网络不用反向传播
            teacher_graph_preds = teacher_model(graph, feat)

        # student model forward
        student_preds = student_model.forward(feat)


        student_loss = cross_entropy(student_preds[train_idx], labels[train_idx])

        ditillation_loss = kl_loss_fn(F.log_softmax(student_preds, dim=-1), F.softmax(teacher_graph_preds, dim=-1))

        loss = args.alpha * student_loss + (1 - args.alpha) * ditillation_loss

        student_optimizer.zero_grad()
        loss.backward()  # 反向传播
        student_optimizer.step()  # 参数优化

        student_model.eval()
        with th.no_grad():
            pred = student_model(feat)
        val_loss = cross_entropy(pred[val_idx], labels[val_idx])
        test_loss = cross_entropy(pred[test_idx], labels[test_idx])

        train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], args.metric,
                                   average=args.average)
        val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], args.metric, average=args.average)
        test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], args.metric, average=args.average)

        wandb.log(
            {'Student_Train_loss': loss, 'Student_Test_loss': test_loss, 'Student_Val_loss': val_loss,
             'Distillation_loss': ditillation_loss, 'Student_Train_Result': train_results,
             'Student_Val_Result': val_results, 'Student_Test_Result': test_results})

        lr_scheduler.step(loss)
        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_result = val_results
            final_test_result = test_results
            if args.save:
                th.save(student_model, filename)

    print('Saving the best student model')
    print('***********************')
    print(f"Best Student Mdeol val acc: {best_val_result}, Final test acc: {final_test_result}")
    return best_val_result, final_test_result


class Classifier(nn.Module):
    def __init__(self, model, in_feats, n_labels):
        super().__init__()
        self.Adapter = model
        hidden_dim = in_feats
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def reset_parameters(self):
        self.Adapter.reset_parameters()

        self.classifier.reset_parameters()

    def forward(self, feat):
        # Extract outputs from the model
        outputs, feat = self.Adapter(feat)
        outputs = outputs + feat
        logits = self.classifier(outputs)
        return logits


class MLP(nn.Module):
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

    def forward(self, feat):
        h = feat

        for i in range(self.n_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)

        return self.linears[-1](h), feat



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
        "--teacher-n-layers", type=int, default=5, help="number of teacher models hidden units"
    )
    argparser.add_argument(
        "--teacher-n-heads", type=int, default=3, help="number of teacher models hidden units"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
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
        "--alpha", type=float, default=0.5, help="learning rate"
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
        "--save_path", type=str, default=None, help="Path to save the Student Model"
    )
    argparser.add_argument(
        "--save", type=bool, default=False, help="Whether to save the student model."
    )
    argparser.add_argument(
        "--teacher_path", type=str, default='/dataintent/local/user/v-haoyan1/Model/', help="Path to save the Teacher Model", required=True
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
    # set_seed(args.seed)
    GraphAdapter = MLP(in_features, n_layers=args.n_layers, n_hidden=args.n_hidden, activation=F.relu,
                       dropout=args.dropout).to(device)
    student_model = Classifier(GraphAdapter, in_feats=in_features, n_labels=n_classes).to(device)

    teacher_model = RevGAT(feat.shape[1], n_classes, args.teacher_n_hidden,  args.teacher_layers, args.teacher_n_heads, F.relu, dropout=0.5, attn_drop=0,
                           edge_drop=0, use_attn_dst=False, use_symmetric_norm=True).to(device)

    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in teacher_model.parameters() if p.requires_grad]
    )
    print(f"Number of the teacher model params: {TRAIN_NUMBERS}")

    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in student_model.parameters() if p.requires_grad]
    )
    print(f"Number of the student model params: {TRAIN_NUMBERS}")


    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = None
    if args.save:
        if not os.path.exists(os.path.dirname(args.save_path)):
            # 创建路径
            os.makedirs(os.path.dirname(args.save_path))

        filename = os.path.join(args.save_path, f"best_student_model_{timestamp}.pkl")
    # First stage, Teacher model pretraining
    # 处理teacher_model 相关的路径文件名
    feature_prefix = os.path.splitext(os.path.basename(args.feature))[0]
    # 创建保存路径
    save_path = os.path.join(args.teacher_path, args.data_name, feature_prefix)
    os.makedirs(save_path, exist_ok=True)
    teacher_file_prefix = f"lr_{args.lr}_h_{args.teacher_n_hidden}_l_{args.teacher_layers}_h_{args.teacher_n_heads}"
    # 保存 teacher model
    model_path = os.path.join(save_path, f"{teacher_file_prefix}.pth")
    teacher_training(args, teacher_model, graph, feat, labels, train_idx, val_idx, test_idx, model_path)


    # run
    val_results = []
    test_results = []

    for run in range(args.n_runs):
        student_model.reset_parameters()
        val_result, test_result = student_training(args, student_model, teacher_model, graph, feat, labels, train_idx, val_idx, test_idx, filename)
        wandb.log({f'Val_{args.metric}': val_result, f'Test_{args.metric}': test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val accuracy: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test accuracy: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})


if __name__ == "__main__":
    main()
