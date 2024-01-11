import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from GraphData import load_data
import torch as th

# 假设你有一个数据集X，其中每一行代表一个样本
gpu = 0
device = th.device("cuda:%d" % gpu if th.cuda.is_available() else 'cpu')

graph, labels, train_idx, val_idx, test_idx = load_data(graph_path=None, name='ogbn-arxiv')

PLM_features = '/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_deberta_base_512_Tuned_cls.npy'
LLM_features = '/dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/Feature/Arxiv_Llama_2_7b_hf_256_mean.npy'

PLM_feat = th.from_numpy(np.load(PLM_features).astype(np.float32))
LLM_feat = th.from_numpy(np.load(LLM_features).astype(np.float32))

sample_size = 1000

# 对 PLM_feat 进行采样和获取标签
PLM_feat_sample = PLM_feat[:sample_size]
PLM_labels_sample = labels[:sample_size]  # 假设你从训练集中采样了样本

# 对 LLM_feat 进行采样和获取标签
LLM_feat_sample = LLM_feat[:sample_size]
LLM_labels_sample = labels[:sample_size]  # 假设你从训练集中采样了样本
pca = PCA(n_components=768)  # 保持与 PLM_feat 相同的维度
LLM_feat_sample_pca = pca.fit_transform(LLM_feat_sample)

# 合并特征矩阵和标签
features = np.vstack((PLM_feat_sample, LLM_feat_sample_pca))
labels = np.concatenate((PLM_labels_sample, LLM_labels_sample))

# 创建 TSNE 对象并进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

# 绘制 t-SNE 可视化结果
plt.scatter(tsne_result[:sample_size, 0], tsne_result[:sample_size, 1], c=PLM_labels_sample, cmap='viridis', label='PLM')
plt.scatter(tsne_result[sample_size:, 0], tsne_result[sample_size:, 1], c=LLM_labels_sample, cmap='viridis', label='LLM')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Combined t-SNE Visualization')
plt.legend()
plt.savefig('combined_tsne.pdf')
plt.show()
