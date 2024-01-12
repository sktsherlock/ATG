import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from GraphData import load_data
import torch as th
import os
import argparse

argparser = argparse.ArgumentParser(
    "TSNE Config",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
argparser.add_argument(
    "--graph_path", type=str, default=None, help="The datasets to be implemented."
)
argparser.add_argument(
    "--save_path", type=str, default='/dataintent/local/user/v-haoyan1/TSNE/OGB/',
    help="The datasets to be implemented."
)
argparser.add_argument(
    "--feat1", type=str,
    default='/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_deberta_base_512_Tuned_cls.npy',
    help="The datasets to be implemented."
)
argparser.add_argument(
    "--feat2", type=str,
    default='/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_deberta_base_512_Tuned_mean.npy',
    help="The datasets to be implemented."
)
argparser.add_argument(
    "--label1", type=str, default='PLM_CLS', help="The datasets to be implemented."
)
argparser.add_argument(
    "--label2", type=str, default='PLM_Mean', help="The datasets to be implemented."
)
args = argparser.parse_args()

# 假设你有一个数据集X，其中每一行代表一个样本
gpu = 0
device = th.device("cuda:%d" % gpu if th.cuda.is_available() else 'cpu')

graph, labels, train_idx, val_idx, test_idx = load_data(graph_path=None, name='ogbn-arxiv')


def visualize(feat1, feat2, path, sample_size=2500, label1='PLM', label2='LLM'):
    # 对 PLM_feat 进行采样和获取标签
    feat1_sample = feat1[:sample_size]
    # 对 LLM_feat 进行采样和获取标签
    feat2_sample = feat2[:sample_size]
    if feat2.shape[1] != feat1.shape[1]:
        pca = PCA(n_components=feat1.shape[1])  # 保持与 PLM_feat 相同的维度
        feat2_sample = pca.fit_transform(feat2_sample)

    # 合并特征矩阵和标签
    features = np.vstack((feat1_sample, feat2_sample))

    # 创建 TSNE 对象并进行降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # 绘制 t-SNE 可视化结果
    plt.scatter(tsne_result[:sample_size, 0], tsne_result[:sample_size, 1], cmap='viridis', label=label1)
    plt.scatter(tsne_result[sample_size:, 0], tsne_result[sample_size:, 1], cmap='viridis', label=label2)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.title(f'T-SNE Visualization between {label1} and {label2}')
    plt.legend()
    save_path = os.path.join(path, f'{label1}_{label2}_combined_tsne.pdf')
    plt.savefig(save_path)
    plt.show()


PLM_cls_features = '/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_deberta_base_512_Tuned_cls.npy'
PLM_mean_features = '/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_deberta_base_512_Tuned_mean.npy'
LLM_7b_features = '/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_Llama_2_7b_hf_256_mean.npy'
LLM_1b_features = '/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_TinyLlama_1.1B_Chat_v1.0_512_mean.npy'

Feature1 = th.from_numpy(np.load(args.feat1).astype(np.float32))
Feature2 = th.from_numpy(np.load(args.feat2).astype(np.float32))

visualize(Feature1, Feature2, args.save_path, label1=args.label1, label2=args.label2)
print('Finished TSNE')