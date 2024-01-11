import dgl
from GraphData import load_data
from collections import Counter
import argparse

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
graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name)

# 获取节点标签
node_labels = labels

# 创建同质性计数器
homophily_counter = Counter()

# 遍历图中的每个节点
for node_id in range(graph.number_of_nodes()):
    # 获取节点的标签
    node_label = node_labels[node_id].item()

    # 获取节点的邻居
    neighbors = graph.successors(node_id)
    neighbor_labels = node_labels[neighbors].tolist()

    # 统计节点与邻居的标签情况
    neighbor_label_counts = Counter(neighbor_labels)

    # 更新同质性计数器
    homophily_counter.update({node_label: neighbor_label_counts[node_label]})

# 打印同质性情况
print("Homophily Counter:")
for label, count in homophily_counter.items():
    print(f"Label {label}: {count} neighbors with the same label")