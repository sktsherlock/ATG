import math
import torch as th
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

epsilon = 1 - math.log(2)


def _contrastive_loss(z1, z2, device, loss_type='simsce'):
    if loss_type == 'simsce':
        return _contrastive_loss_simsce(z1, z2, device=device)
    else:
        raise ValueError('Error contrastive loss type {}!'.format(loss_type))


def _contrastive_loss_simsce(z1, z2, device, similarity='inner', temperature=0.1):
    assert z1.shape == z2.shape
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    if similarity == 'inner':
        similarity_matrix = th.matmul(z1, z2.T)
    elif similarity == 'cosine':
        similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
    else:
        similarity_matrix = th.matmul(z1, z2.T)
    similarity_matrix /= temperature

    labels = th.arange(similarity_matrix.shape[0]).long().to(device)
    loss_res = F.cross_entropy(similarity_matrix, labels, reduction="mean")
    return loss_res


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, loss):
        score = loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop



def adjust_learning_rate(optimizer, lr, epoch, warmup_epochs=50):
    if epoch <= warmup_epochs:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / warmup_epochs


def cross_entropy(x, target, label_smoothing=0.0):
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
