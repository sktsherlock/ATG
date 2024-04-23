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
parser.add_argument("--text_logits", type=str, default='Exp/Transductive/Movies/GCN/TextFeature/', help="The logits generated from the text feature for ensembling")
parser.add_argument("--visual_logits", type=str, default='Exp/Transductive/Movies/GCN/ImageFeature/', help="The logits generated from the visual feature for ensembling")
parser.add_argument("--graph_path", type=str, default=None, help="The datasets to be implemented.")
parser.add_argument("--c_and_s", action="store_true", help="correct and smoothing")
parser.add_argument("--text_feature_weight", default=1.0, type=float)
parser.add_argument("--image_feature_weight", default=1.0, type=float)
parser.add_argument("--start_seed", type=int, default=42)
parser.add_argument(
    "--fewshots", type=int, default=None, help="fewshots values"
)
parser.add_argument(
    "--data_name", type=str, default=None, help="The dataset name.",
)
parser.add_argument(
    "--train_ratio", type=float, default=0.6, help="training ratio"
)
parser.add_argument(
    "--val_ratio", type=float, default=0.2, help="training ratio"
)
parser.add_argument(
    "--metric", type=str, default='accuracy', choices=['accuracy', 'precision', 'recall', 'f1'],
    help="The metric to be used."
)
parser.add_argument(
    "--average", type=str, default=None, choices=['weighted', 'micro', 'macro', None]
)




def ensembling(text_feature_path, image_feature_path, args, labels, train_idx, val_idx, test_idx, c_and_s=False):
    # 加载预测文件
    text_feature_pred = np.load(text_feature_path)
    image_feature_pred = np.load(image_feature_path)
    y_pred = (args.text_feature_weight * text_feature_pred + args.image_feature_weight * image_feature_pred) / (args.text_feature_weight + args.image_feature_weight)

    # if c_and_s:
    #     y_pred = correct_and_smooth(data, split_idx, y_pred)

    # y_pred = y_pred.argmax(dim=-1, keepdim=True)
    y_true = labels
    train_results = get_metric(torch.argmax(y_pred[train_idx], dim=1), y_true[train_idx], metric, average=average)
    valid_results = get_metric(torch.argmax(y_pred[val_idx], dim=1), y_true[val_idx], metric, average=average)
    test_results = get_metric(torch.argmax(y_pred[test_idx], dim=1), y_true[test_idx], metric, average=average)

    return train_results, valid_results, test_results


def compute():
    args = parser.parse_args()
    # text_logits = args.text_logits + args.metric
    # visual_logits = args.visual_logits + args.metric

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(root_dir.rstrip('/')))
    text_logits = os.path.join(base_dir, args.text_logits, args.metric)
    visual_logits = os.path.join(base_dir, args.visual_logits, args.metric)
    graph_path = os.path.join(base_dir, args.graph_path)
    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name,
                                                            fewshots=args.fewshots)
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for seed in range(args.start_seed, args.start_seed + 10):
        text_feature_path = f'{text_logits}/Seed{seed}/Seed{seed}.npy'
        image_feature_path = f'{visual_logits}/Seed{seed}/Seed{seed}.npy'
        train_acc, val_acc, test_acc = ensembling(text_feature_path, image_feature_path, args, labels, train_idx, val_idx, test_idx, c_and_s=args.c_and_s)
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