import numpy as np

text_features = np.load('textfeature.npy')  # 加载文本特征文件
image_features = np.load('imagefeature.npy')  # 加载图像特征文件

combined_features = np.concatenate((text_features, image_features), axis=1)  # 合并特征文件

np.save('combined_features.npy', combined_features)  # 保存新特征文件