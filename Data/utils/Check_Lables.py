from collections import Counter
import dgl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--graph_path', type=str, help='Path to the graph file', required=True)
args = parser.parse_args()


graph = dgl.load_graphs(args.graph_path)[0][0]

labels = graph.ndata['label'].tolist()
label_counts = Counter(labels)

for label, count in label_counts.items():
    print(f"Label {label}: {count}")
