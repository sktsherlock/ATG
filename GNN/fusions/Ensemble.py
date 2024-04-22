import argparse
import sys
import os
import numpy as np
import torch
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.transforms import ToSparseTensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GraphData import load_data
from LossFunction import get_metric

parser = argparse.ArgumentParser()
parser.add_argument("--list_logits", type=str, help="for ensembling")
parser.add_argument("--graph_path", type=str, default=None, help="The datasets to be implemented.")
parser.add_argument("--c_and_s", action="store_true", help="correct and smoothing")
parser.add_argument("--weights", nargs="+", type=float)
parser.add_argument("--start_seed", type=int, default=42)
parser.add_argument(
    "--metric", type=str, default='accuracy', choices=['accuracy', 'precision', 'recall', 'f1'],
    help="The metric to be used."
)
parser.add_argument(
    "--average", type=str, default=None, choices=['weighted', 'micro', 'macro', None]
)




def ensembling(list_logits, c_and_s=False):

    list_logits = [torch.load(logits).cpu() for logits in list_logits]
    weights = np.asarray(args.weights) / sum(args.weights)
    list_logits = [
        logits.softmax(dim=-1) * weight for logits, weight in zip(list_logits, weights)
    ]
    y_pred = sum(list_logits) / len(list_logits)

    if c_and_s:
        y_pred = correct_and_smooth(data, split_idx, y_pred)

    # y_pred = y_pred.argmax(dim=-1, keepdim=True)
    y_true = labels
    train_results = get_metric(th.argmax(y_pred[train_idx], dim=1), y_true[train_idx], metric, average=average)
    valid_results = get_metric(th.argmax(y_pred[val_idx], dim=1), y_true[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(y_pred[test_idx], dim=1), y_true[test_idx], metric, average=average)

    return train_results, valid_results, test_results


def compute():
    args = parser.parse_args()

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name,
                                                            fewshots=args.fewshots)
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for seed in range(args.start_seed, args.start_seed + 10):
        list_logits = args.list_logits.split(" ")
        list_logits = [logits + f"/logits_seed{seed}.pt" for logits in list_logits]
        train_acc, val_acc, test_acc = ensembling(list_logits, c_and_s=args.c_and_s)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    print(
        "train_acc: {:.2f} ± {:.2f}".format(
            100 * np.mean(train_acc_list), 100 * np.std(train_acc_list)
        )
    )
    print(
        "val_acc: {:.2f} ± {:.2f}".format(
            100 * np.mean(val_acc_list), 100 * np.std(val_acc_list)
        )
    )
    print(
        "test_acc: {:.2f} ± {:.2f}".format(
            100 * np.mean(test_acc_list), 100 * np.std(test_acc_list)
        )
    )


def correct_and_smooth(data, split_idx, y_soft):
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    if not hasattr(data, "adj_t"):
        data = ToSparseTensor()(data)
    adj_t = data.adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

    y, train_idx = data.y.to(device), split_idx["train"].to(device)
    y_train = y[train_idx]
    y_soft = y_soft.to(device)

    post = CorrectAndSmooth(
        num_correction_layers=50,
        correction_alpha=0.5,
        num_smoothing_layers=50,
        smoothing_alpha=0.5,
        autoscale=False,
        scale=20.0,
    )

    print("Correct and smooth...")
    y_soft = post.correct(y_soft, y_train, train_idx, DAD)
    y_soft = post.smooth(y_soft, y_train, train_idx, DA)
    print("Done!")

    return y_soft


if __name__ == "__main__":
    compute()