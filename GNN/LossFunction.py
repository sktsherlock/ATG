import math
import torch as th
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

epsilon = 1 - math.log(2)


def cross_entropy(x, target, label_smoothing):
    y = F.cross_entropy(x, target, reduction="mean", label_smoothing=label_smoothing)
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def get_metric(y_true, y_pred, metric, average=None):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred, average=average)
    elif metric == "recall":
        return recall_score(y_true, y_pred, average=average)
    elif metric == "f1":
        return f1_score(y_true, y_pred, average=average)
    else:
        raise ValueError("Unsupported metric: {}".format(metric))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Metric debug",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--average", type=str, default=None, choices=['weighted', 'micro', 'macro', None]
    )
    args = argparser.parse_args()

    pred = [1, 0, 1, 1, 0]
    labels = [1, 0, 0, 1, 1]

    accuracy = get_metric(pred, labels, "accuracy")
    precision = get_metric(pred, labels, "precision")
    recall = get_metric(pred, labels, "recall")
    f1 = get_metric(pred, labels, "f1", average=args.average)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print(f"F1 in {args.average}:", f1)
