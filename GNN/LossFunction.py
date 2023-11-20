import math
import torch as th
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

epsilon = 1 - math.log(2)


def cross_entropy(x, target, label_smoothing):
    y = F.cross_entropy(x, target, reduction="mean", label_smoothing=label_smoothing)
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def get_metric(y_true, y_pred, metric, average='weighted'):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred)
    elif metric == "recall":
        return recall_score(y_true, y_pred)
    elif metric == "f1":
        return f1_score(y_true, y_pred, average=average)
    else:
        raise ValueError("Unsupported metric: {}".format(metric))


if __name__ == "__main__":
    pred = [1, 0, 1, 1, 0]
    labels = [1, 0, 0, 1, 1]

    accuracy = get_metric(pred, labels, "accuracy")
    precision = get_metric(pred, labels, "precision")
    recall = get_metric(pred, labels, "recall")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
