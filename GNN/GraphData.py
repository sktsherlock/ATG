import numpy as np
import torch as th
import dgl
import os
import random
from ogb.nodeproppred import DglNodePropPredDataset
import wandb
from torch_geometric.utils import from_dgl

def split_graph(nodes_num, train_ratio, val_ratio, labels, fewshots=None):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    if fewshots is not None:
        train_ids = []

        unique_labels = np.unique(labels)  # 获取唯一的类别标签
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]  # 获取属于当前类别的样本索引
            np.random.shuffle(label_indices)  # 对当前类别的样本索引进行随机排序

            fewshot_indices = label_indices[:fewshots]  # 选择指定数量的few-shot样本
            train_ids.extend(fewshot_indices)

        remaining_indices = np.setdiff1d(indices, train_ids)  # 获取剩余的样本索引
        np.random.shuffle(remaining_indices)  # 对剩余样本索引进行随机排序

        val_size = int(len(remaining_indices) * val_ratio)  # 计算验证集大小

        val_ids = remaining_indices[:val_size]  # 划分验证集
        test_ids = remaining_indices[val_size:]  # 划分测试集

    else:

        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_ids = indices[:train_size]
        val_ids = indices[train_size:train_size + val_size]
        test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


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


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def split_edge(graph, test_ratio=0.2, val_ratio=0.1, random_seed=42, neg_len='1000', path=None):
    if os.path.exists(os.path.join(path, f'{neg_len}/edge_split.pt')):
        edge_split = th.load(os.path.join(path, f'{neg_len}/edge_split.pt'))

    else:

        np.random.seed(random_seed)
        th.manual_seed(random_seed)

        eids = np.arange(graph.num_edges())
        eids = np.random.permutation(eids)

        all_source, all_target = graph.edges()
        # all_source = graph.edge_index[0]
        # all_target = graph.edge_index[1]
        test_size = int(len(eids) * test_ratio)
        val_size = int(len(eids) * val_ratio)


        test_source, test_target = all_source[eids[:test_size]], all_target[eids[:test_size]]
        val_source, val_target = all_source[eids[test_size:test_size + val_size]], all_target[eids[test_size:test_size + val_size]]
        tra_source, tra_target = all_source[eids[test_size + val_size:]], all_target[eids[test_size + val_size:]]


        val_target_neg = th.randint(low=0, high=graph.num_nodes(), size=(len(val_source), int(neg_len)))
        test_target_neg = th.randint(low=0, high=graph.num_nodes(), size=(len(test_source), int(neg_len)))

        # ! 创建dict类型存法
        edge_split = {'train': {'source_node': tra_source, 'target_node': tra_target},
                      'valid': {'source_node': val_source, 'target_node': val_target,
                                'target_node_neg': val_target_neg},
                      'test': {'source_node': test_source, 'target_node': test_target,
                               'target_node_neg': test_target_neg}}

        th.save(edge_split, os.path.join(path, 'edge_split.pt'))

    return edge_split



def _eval_mrr(y_pred_pos, y_pred_neg, type_info):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''


    if type_info == 'torch':
        # calculate ranks
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits1_list = (ranking_list <= 1).to(th.float)
        hits5_list = (ranking_list <= 5).to(th.float)
        hits10_list = (ranking_list <= 10).to(th.float)
        mrr_list = 1./ranking_list.to(th.float)

        return {'hits@1_list': hits1_list,
                 'hits@5_list': hits5_list,
                 'hits@10_list': hits10_list,
                 'mrr_list': mrr_list}

    else:
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits1_list = (ranking_list <= 1).astype(np.float32)
        hits5_list = (ranking_list <= 5).astype(np.float32)
        hits10_list = (ranking_list <= 10).astype(np.float32)
        mrr_list = 1./ranking_list.astype(np.float32)

        return {'hits@1_list': hits1_list,
                 'hits@5_list': hits5_list,
                 'hits@10_list': hits10_list,
                 'mrr_list': mrr_list}


class Evaluator:
    def __init__(self):
        self.K = 10


    def _parse_and_check_input(self, input_dict):

        if not 'y_pred_pos' in input_dict:
            raise RuntimeError('Missing key of y_pred_pos')
        if not 'y_pred_neg' in input_dict:
            raise RuntimeError('Missing key of y_pred_neg')

        y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

        '''
            y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )
            y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, )
        '''

        # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
        # type_info stores information whether torch or numpy is used

        type_info = None

        # check the raw tyep of y_pred_pos
        if not (isinstance(y_pred_pos, np.ndarray) or (th is not None and isinstance(y_pred_pos, th.Tensor))):
            raise ValueError('y_pred_pos needs to be either numpy ndarray or th tensor')

        # check the raw type of y_pred_neg
        if not (isinstance(y_pred_neg, np.ndarray) or (th is not None and isinstance(y_pred_neg, th.Tensor))):
            raise ValueError('y_pred_neg needs to be either numpy ndarray or th tensor')

        # if either y_pred_pos or y_pred_neg is th tensor, use th tensor
        if th is not None and (isinstance(y_pred_pos, th.Tensor) or isinstance(y_pred_neg, th.Tensor)):
            # converting to th.Tensor to numpy on cpu
            if isinstance(y_pred_pos, np.ndarray):
                y_pred_pos = th.from_numpy(y_pred_pos)

            if isinstance(y_pred_neg, np.ndarray):
                y_pred_neg = th.from_numpy(y_pred_neg)

            # put both y_pred_pos and y_pred_neg on the same device
            y_pred_pos = y_pred_pos.to(y_pred_neg.device)

            type_info = 'torch'

        else:
            # both y_pred_pos and y_pred_neg are numpy ndarray

            type_info = 'numpy'

        if not y_pred_pos.ndim == 1:
            raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

        if not y_pred_neg.ndim == 1:
            raise RuntimeError('y_pred_neg must to 1-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

        return y_pred_pos, y_pred_neg, type_info



    def eval(self, input_dict):
        y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
        return _eval_mrr(y_pred_pos, y_pred_neg, type_info)


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, key='Hits@10'):
        if run is not None:
            result = 100 * th.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * th.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = th.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train {key}: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log({f'Highest_Train_{key}': float(f'{r.mean():.2f}'),
                       f'Highest_Train_{key}_Std': float(f'{r.std():.2f}')})
            r = best_result[:, 1]
            print(f'Highest_Valid_{key}: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log({f'{key} Highest Valid Acc': float(f'{r.mean():.2f}'),
                       f'{key} Highest Valid Std': float(f'{r.std():.2f}')})
            r = best_result[:, 2]
            print(f'{key} Final Train: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log(
                {f'{key} Final Train Acc': float(f'{r.mean():.2f}'), f'{key} Final Train Std': float(f'{r.std():.2f}')})
            r = best_result[:, 3]
            print(f'{key} Final Test: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log(
                {f'{key} Final Test Acc': float(f'{r.mean():.2f}'), f'{key} Final Test Std': float(f'{r.std():.2f}')})