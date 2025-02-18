from tqdm import  tqdm
from collections import Counter
import argparse
import torch as th
import dgl
from ogb.nodeproppred import DglNodePropPredDataset


def load_data(graph_path, train_ratio=0.6, val_ratio=0.2, name=None, fewshots=None,):
    if name == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=name)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        labels = labels[:, 0]
    else:
        # load the graph from local path
        graph = dgl.load_graphs(graph_path)[0][0]
        labels = graph.ndata['label']
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), train_ratio, val_ratio, labels, fewshots=fewshots)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    return graph, labels, train_idx, val_idx, test_idx


argparser = argparse.ArgumentParser(
    "Homo Config",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
argparser.add_argument(
    "--graph_path", type=str, default=None, help="The datasets to be implemented."
)
argparser.add_argument(
    "--train_ratio", type=float, default=0.6, help="training ratio"
)
argparser.add_argument(
    "--val_ratio", type=float, default=0.2, help="training ratio"
)
argparser.add_argument(
    "--data_name", type=str, default=None, help="The dataset name.",
)
args = argparser.parse_args()

# 假设已加载了图数据集，存储在变量graph中
graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                        val_ratio=args.val_ratio, name=args.data_name)

# 获取节点标签
node_labels = labels

# 创建同质性计数器
homophily_counter = Counter()

# 计算标签相同的邻居节点边的数量
num_homophily_edges = 0
total_edges = graph.number_of_edges()

for edge in tqdm(range(total_edges)):
    src, dst = graph.find_edges(edge)
    src_label = labels[src]
    dst_label = labels[dst]
    if src_label == dst_label:
        num_homophily_edges += 1

# 计算同质性比率
homophily_ratio = num_homophily_edges / total_edges

print("Homophily Ratio:", homophily_ratio)
