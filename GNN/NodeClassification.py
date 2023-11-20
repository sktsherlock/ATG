import time
import wandb
import numpy as np
import torch as th
import torch.optim as optim
from LossFunction import cross_entropy, get_metric


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

    train_results = get_metric(pred[train_idx], labels[train_idx], metric, average=average)
    val_results = get_metric(pred[val_idx], labels[val_idx], metric, average=average)
    test_results = get_metric(pred[test_idx,], labels[test_idx], metric, average=average)

    return train_results, val_results, test_results, val_loss, test_loss




def classification(
        args, graph, model, feat, labels, train_idx, val_idx, test_idx, n_running):
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
            wandb.log({'Train_loss': train_loss, 'Val_loss': val_loss, 'Test_loss': test_loss})
            lr_scheduler.step(train_loss)

            toc = time.time()
            total_time += toc - tic

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_result = val_result
                final_test_result = test_result

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                    f"Loss: {train_loss.item():.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
                )

    print("*" * 50)
    print(f"Best val acc: {best_val_result}, Final test acc: {final_test_result}")
    print("*" * 50)

    return best_val_result, final_test_result
