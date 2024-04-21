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

from fusions.MultiBench import TensorFusion
from GraphData import load_data, set_seed
from NodeClassification import mag_classification
from Utils.model_config import add_common_args, add_revgat_args, add_sage_args, add_gat_args, add_sgc_args, add_appnp_args, gen_model, gen_fusion, add_fusion_args



class MAGNN(nn.Module):
    def __init__(self, early_fusion, gnn):
        super().__init__()
        self.early_fusion = early_fusion
        self.gnn = gnn

    def forward(self, graph, text_feature, visual_feature):
        feat = self.early_fusion(text_feature, visual_feature)
        h = self.gnn(graph, feat)
        return h


# 参数定义模块
def args_init():
    argparser = argparse.ArgumentParser(
        "MAGNN Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    add_sgc_args(argparser)
    add_appnp_args(argparser)
    add_sage_args(argparser)
    add_gat_args(argparser)
    add_revgat_args(argparser)
    add_fusion_args(argparser)
    return argparser


def main():
    argparser = args_init()
    args = argparser.parse_args()
    wandb.init(config=args, reinit=True)

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() else 'cpu')

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name, fewshots=args.fewshots)

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

    # ! 加载多模态特征
    Text_feature = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    Visual_feature = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

    n_classes = (labels.max()+1).item()
    print(f"Number of classes {n_classes}, Number of text features {Text_feature.shape[1]}, Number of visual features {Visual_feature.shape[1]}")

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)

    # run
    val_results = []
    test_results = []


    for run in range(args.n_runs):
        set_seed(args.seed + run)
        # Model implementation
        GNN = gen_model(args, device, n_classes)
        Fusion = gen_fusion(args, device)

        # MAGNN implementation
        Model = MAGNN(Fusion, GNN)
        TRAIN_NUMBERS = sum(
            [np.prod(p.size()) for p in Model.parameters() if p.requires_grad]
        )
        print(f"Number of the all MAGNN model params: {TRAIN_NUMBERS}")
        Model.reset_parameters()
        val_result, test_result = mag_classification(
            args, graph, observe_graph, Model, Text_feature, Visual_feature, labels, train_idx, val_idx, test_idx, run+1
        )
        wandb.log({f'Val_{args.metric}': val_result, f'Test_{args.metric}': test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val {args.metric}: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test {args.metric}: {np.mean(test_results)} ± {np.std(test_results)}")
    wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})


if __name__ == "__main__":
    main()
