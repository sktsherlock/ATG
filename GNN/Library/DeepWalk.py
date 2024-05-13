import sys
import copy
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import argparse
import wandb
import torch as th
import numpy as np
import torch.nn.functional as F
import os
from dgl.sampling import node2vec_random_walk
from sklearn.linear_model import LogisticRegression
from dgl.nn.pytorch import DeepWalk
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LossFunction import get_metric
from GraphData import load_data, set_seed
from Utils.model_config import add_common_args


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for batch_walk in loader:
        batch_walk = batch_walk.to(device)
        loss = model(batch_walk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@th.no_grad()
def evaluate(model, labels, train_idx, val_idx, test_idx, metric, average):
    model.eval()
    X = model.node_embed.weight.detach()

    lr = LogisticRegression(
        solver="lbfgs", multi_class="auto", max_iter=500
    ).fit(X[train_idx].numpy, labels[train_idx].numpy())

    train_results = get_metric(lr.predict(X[train_idx]), labels[train_idx], metric, average=average)
    val_results = get_metric(lr.predict(X[val_idx]), labels[val_idx], metric, average=average)
    test_results = get_metric(lr.predict(X[test_idx]), labels[test_idx], metric, average=average)

    return train_results, val_results, test_results


# 参数定义模块
def args_init():
    argparser = argparse.ArgumentParser(
        "Deepwalk Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    return argparser


def main():
    argparser = args_init()
    args = argparser.parse_args()
    wandb.init(config=args, reinit=True)

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() else 'cpu')

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name,
                                                            fewshots=args.fewshots)

    # add reverse edges, tranfer to the  undirected graph
    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    # 定义可观测图数据，用于inductive实验设置；
    observe_graph = copy.deepcopy(graph)

    if args.inductive:
        # 构造Inductive Learning 实验条件

        isolated_nodes = th.cat((val_idx, test_idx))
        sort_isolated_nodes, _ = th.sort(isolated_nodes)
        # 从图中删除指定节点
        observe_graph.remove_nodes(sort_isolated_nodes)

        # 添加相同数量的孤立节点
        observe_graph.add_nodes(len(sort_isolated_nodes))
        print(observe_graph)
        print('***************')
        print(graph)

    # add self-loop
    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    # feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device) if args.feature is not None else graph.ndata['feat'].to(device)
    n_classes = (labels.max() + 1).item()
    # print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    graph.create_formats_()
    observe_graph.create_formats_()

    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    # run
    val_results = []
    test_results = []

    # Model implementation
    # model = GCN(feat.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)



    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model = DeepWalk(observe_graph).to(device)
        model.reset_parameters()
        loader = DataLoader(th.arange(observe_graph.num_nodes()), batch_size=128,
                            shuffle=True, collate_fn=model.sample)
        optimizer = th.optim.SparseAdam(model.parameters(), lr=args.lr)

        best_val_result, final_test_result = 0, 0

        for epoch in range(1, args.n_epochs + 1):
            train_loss = train(model, loader, optimizer, device)

            if epoch % args.eval_steps == 0:
                train_result, val_result, test_result = evaluate(model, labels, train_idx, val_idx, test_idx,
                                                                 args.metric, args.average)
                wandb.log(
                    {'Train_loss': train_loss,
                     'Train_result': train_result,
                     'Val_result': val_result, 'Test_result': test_result})

                if val_result > best_val_result:
                    best_val_result = val_result
                    final_test_result = test_result

        # val_result, test_result = classification(
        #     args, graph, observe_graph, model, feat, labels, train_idx, val_idx, test_idx, run+1
        # )

        wandb.log({f'Val_{args.metric}': best_val_result, f'Test_{args.metric}': final_test_result})
        val_results.append(best_val_result)
        test_results.append(final_test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val {args.metric}: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test {args.metric}: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})


if __name__ == "__main__":
    main()
