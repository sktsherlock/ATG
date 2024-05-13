import dgl
import argparse
# from ge.classify import read_node_label,
from gensim.models import Word2Vec
from walker import RandomWalker, Classifier
from sklearn.linear_model import LogisticRegression
import networkx as nx
import sys
import os
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.model_config import add_common_args
from GraphData import load_data, set_seed


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


def args_init():
    argparser = argparse.ArgumentParser(
        "DeepWalk Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    return argparser



def evaluate_embeddings(x, y, tid, vid, testid):
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    val_result, test_result = clf.train_evaluate(x, y, tid, vid, testid)
    return val_result, test_result




if __name__ == "__main__":
    argparser = args_init()
    args = argparser.parse_args()
    wandb.init(config=args, reinit=True)
    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name, fewshots=args.fewshots)

    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    # add self-loop
    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    nx_g = dgl.to_networkx(graph)

    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    for run in range(args.n_runs):

        set_seed(args.seed + run)

        model = DeepWalk(nx_g, walk_length=10, num_walks=80, workers=1)
        model.train(window_size=1, iter=1) # 5,3
        embeddings = model.get_embeddings()

        val_result, test_result = evaluate_embeddings(embeddings, labels, train_idx, val_idx, test_idx)
