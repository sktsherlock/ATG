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



# 创建TSNE对象并进行降维
tsne = TSNE(n_components=2, random_state=42)
PLM_tsne = tsne.fit_transform(PLM_feat)
LLM_tsne = tsne.fit_transform(LLM_feat)

# 绘制 PLM_feat 的 t-SNE 结果
plt.scatter(PLM_tsne[:, 0], PLM_tsne[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('PLM_feat t-SNE Visualization')
plt.savefig('PLM_feat_tsne.pdf')
plt.close()

# 绘制 LLM_feat 的 t-SNE 结果
plt.scatter(LLM_tsne[:, 0], LLM_tsne[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('LLM_feat t-SNE Visualization')
plt.savefig('LLM_feat_tsne.pdf')
plt.close()