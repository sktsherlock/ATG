import time
import wandb
import os
import torch as th
import torch.optim as optim
from GNN.Utils.LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate


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


import os
import time
import torch as th
import wandb


def classification(args, graph, observe_graph, model, feat, labels, train_idx, val_idx, test_idx, n_running, save_path=None):
    stopper = initialize_early_stopping(args)
    optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

    total_time = 0
    best_val_result, final_test_result, best_val_loss = 0, 0, float("inf")

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        adjust_learning_rate_if_needed(args, optimizer, epoch)

        train_loss, pred = train_model(model, observe_graph, feat, labels, train_idx, optimizer, args)

        if epoch % args.eval_steps == 0:
            train_result, val_result, test_result, val_loss, test_loss = evaluate_model(
                args, model, graph, feat, labels, train_idx, val_idx, test_idx
            )
            log_results_to_wandb(train_loss, val_loss, test_loss, train_result, val_result, test_result)
            lr_scheduler.step(train_loss)

            total_time += time.time() - tic

            if val_result > best_val_result:
                best_val_result, final_test_result = update_best_results(val_result, test_result, save_path, model)

            if should_early_stop(stopper, val_result):
                break

            log_progress(args, epoch, n_running, total_time, train_loss, val_loss, test_loss, train_result, val_result, test_result, best_val_result, final_test_result)

    print_final_results(best_val_result, final_test_result, args)

    if save_path is not None:
        pred = infer_model(graph, feat, save_path)
        return best_val_result, final_test_result, pred

    return best_val_result, final_test_result


def initialize_early_stopping(args):
    return EarlyStopping(patience=args.early_stop_patience) if args.early_stop_patience else None


def initialize_optimizer_and_scheduler(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=args.min_lr)
    return optimizer, lr_scheduler


def adjust_learning_rate_if_needed(args, optimizer, epoch):
    if args.warmup_epochs:
        adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)


def train_model(model, observe_graph, feat, labels, train_idx, optimizer, args):
    return train(model, observe_graph, feat, labels, train_idx, optimizer, label_smoothing=args.label_smoothing)


def evaluate_model(args, model, graph, feat, labels, train_idx, val_idx, test_idx):
    return evaluate(model, graph, feat, labels, train_idx, val_idx, test_idx, args.metric, args.label_smoothing, args.average)


def log_results_to_wandb(train_loss, val_loss, test_loss, train_result, val_result, test_result):
    wandb.log({
        'Train_loss': train_loss, 'Val_loss': val_loss, 'Test_loss': test_loss,
        'Train_result': train_result, 'Val_result': val_result, 'Test_result': test_result
    })


def update_best_results(val_result, test_result, save_path, model):
    best_val_result = val_result
    final_test_result = test_result
    if save_path is not None:
        th.save(model, os.path.join(save_path, "model.pt"))
    return best_val_result, final_test_result


def should_early_stop(stopper, val_result):
    return stopper and stopper.step(val_result)


def log_progress(args, epoch, n_running, total_time, train_loss, val_loss, test_loss, train_result, val_result, test_result, best_val_result, final_test_result):
    if epoch % args.log_every == 0:
        print(
            f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
            f"Loss: {train_loss:.4f}\n"
            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
            f"Train/Val/Test/Best Val/Final Test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
        )


def print_final_results(best_val_result, final_test_result, args):
    print(f"{'*' * 50}\nBest val {args.metric}: {best_val_result}, Final test {args.metric}: {final_test_result}\n{'*' * 50}")


def infer_model(graph, feat, save_path):
    load_model = th.load(os.path.join(save_path, "model.pt"))
    load_model.eval()
    with th.no_grad():
        pred = load_model(graph, feat)
    print('The prediction files is made.')
    return pred


def mag_train(model, graph, text_feat, visual_feat, labels, train_idx, optimizer, label_smoothing):
    model.train()

    optimizer.zero_grad()
    pred = model(graph, text_feat, visual_feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def mag_evaluate(
        model, graph, text_feat, visual_feat, labels, train_idx, val_idx, test_idx, metric='accuracy',
        label_smoothing=0.1, average=None
):
    model.eval()
    with th.no_grad():
        pred = model(graph, text_feat, visual_feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing)
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing)

    train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], metric, average=average)
    val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], metric, average=average)

    return train_results, val_results, test_results, val_loss, test_loss


def mag_classification(
        args, graph, observe_graph, model, text_feat, visual_feat, labels, train_idx, val_idx, test_idx, n_running):
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

        train_loss, pred = mag_train(
            model, observe_graph, text_feat, visual_feat, labels, train_idx, optimizer,
            label_smoothing=args.label_smoothing
        )
        if epoch % args.eval_steps == 0:
            (
                train_result,
                val_result,
                test_result,
                val_loss,
                test_loss,
            ) = mag_evaluate(
                model,
                graph,
                text_feat,
                visual_feat,
                labels,
                train_idx,
                val_idx,
                test_idx,
                args.metric,
                args.label_smoothing,
                args.average
            )
            wandb.log(
                {'Train_loss': train_loss, 'Val_loss': val_loss, 'Test_loss': test_loss, 'Train_result': train_result,
                 'Val_result': val_result, 'Test_result': test_result})
            lr_scheduler.step(train_loss)

            toc = time.time()
            total_time += toc - tic

            if val_result > best_val_result:
                best_val_result = val_result
                final_test_result = test_result

            if args.early_stop_patience is not None:
                if stopper.step(val_result):
                    break

            if epoch % args.log_every == 0:
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                    f"Loss: {train_loss.item():.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best Val/Final Test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
                )

    print("*" * 50)
    print(f"Best val  {args.metric}: {best_val_result}, Final test  {args.metric}: {final_test_result}")
    print("*" * 50)

    return best_val_result, final_test_result
