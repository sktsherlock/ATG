import sys
import copy
import torch.nn as nn
import argparse
import wandb
import torch as th
import numpy as np
import os
from dgl.sampling import node2vec_random_walk
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GNN.Utils.LossFunction import get_metric
from ..GraphData import load_data, set_seed
from ..Utils.model_config import add_common_args


# 模型定义模块
class Node2vec(nn.Module):
    """Node2vec model from paper node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.

        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.

        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(
        self,
        g,
        embedding_dim,
        walk_length,
        p,
        q,
        num_walks=10,
        window_size=5,
        num_negatives=5,
        use_sparse=True,
        weight_name=None,
    ):
        super(Node2vec, self).__init__()

        assert walk_length >= window_size

        self.g = g
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.N = self.g.num_nodes() # 可能出错
        if weight_name is not None:
            self.prob = weight_name
        else:
            self.prob = None

        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def sample(self, batch):
        """
        Generate positive and negative samples.
        Positive samples are generated from random walk
        Negative samples are generated from random sampling
        """
        if not isinstance(batch, th.Tensor):
            batch = th.tensor(batch)

        batch = batch.repeat(self.num_walks)
        # positive
        pos_traces = node2vec_random_walk(
            self.g, batch, self.p, self.q, self.walk_length, self.prob
        )
        pos_traces = pos_traces.unfold(1, self.window_size, 1)  # rolling window
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        # negative
        neg_batch = batch.repeat(self.num_negatives)
        neg_traces = th.randint(
            self.N, (neg_batch.size(0), self.walk_length)
        )
        neg_traces = th.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        neg_traces = neg_traces.unfold(1, self.window_size, 1)  # rolling window
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return pos_traces, neg_traces

    def forward(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.

        Returns
        -------
        Tensor
            Node embedding

        """
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, pos_trace, neg_trace):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
        neg_trace: Tensor
            negative random walk trace

        """
        e = 1e-15

        # Positive
        pos_start, pos_rest = (
            pos_trace[:, 0],
            pos_trace[:, 1:].contiguous(),
        )  # start node and following trace
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # Negative
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)
        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss
        pos_loss = -th.log(th.sigmoid(pos_out) + e).mean()
        neg_loss = -th.log(1 - th.sigmoid(neg_out) + e).mean()

        return pos_loss + neg_loss

    def loader(self, batch_size):
        """

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        DataLoader
            Node2vec training data loader

        """
        return DataLoader(
            th.arange(self.N),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.sample,
        )


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for pos_traces, neg_traces in loader:
        pos_traces, neg_traces = pos_traces.to(device), neg_traces.to(
            device
        )
        optimizer.zero_grad()
        loss = model.loss(pos_traces, neg_traces)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@th.no_grad()
def evaluate(model, labels, train_idx, val_idx, test_idx, metric, average):
    model.eval()
    with th.no_grad():
        x_train = model(train_idx)
        x_val = model(val_idx)
        x_test = model(test_idx)
    x_train = x_train.cpu().numpy()
    x_val = x_val.cpu().numpy()
    x_test = x_test.cpu().numpy()

    lr = LogisticRegression(
        solver="lbfgs", multi_class="auto", max_iter=500
    ).fit(x_train, labels[train_idx])

    train_results = get_metric(lr.predict(x_train), labels[train_idx], metric, average=average)
    val_results = get_metric(lr.predict(x_val), labels[val_idx], metric, average=average)
    test_results = get_metric(lr.predict(x_test), labels[test_idx], metric, average=average)

    return train_results, val_results, test_results


# 参数定义模块
def args_init():
    argparser = argparse.ArgumentParser(
        "Node2Vec Config",
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
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio, val_ratio=args.val_ratio, name=args.data_name, fewshots=args.fewshots)

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
    n_classes = (labels.max()+1).item()
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
    model = Node2vec(observe_graph, 128, 5, 0.25, 4.0, 10, 5, 5, False, None).to(device)
    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the Node2Vec model params: {TRAIN_NUMBERS}")

    for run in range(args.n_runs):
        set_seed(args.seed + run)


        model.reset_parameters()
        loader = model.loader(batch_size=128)
        optimizer = th.optim.AdamW(model.parameters(), lr=args.lr)

        best_val_result, final_test_result = 0, 0

        for epoch in range(1, args.n_epochs + 1):
            train_loss = train(model, loader, optimizer, device)

            if epoch % args.eval_steps == 0:
                train_result, val_result, test_result = evaluate(model, labels, train_idx, val_idx, test_idx, args.metric, args.average)
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