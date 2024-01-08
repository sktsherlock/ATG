import argparse
import numpy as np
from sklearn.decomposition import PCA


def reduce_dimension(features, n_components):
    # 创建PCA模型
    pca = PCA(n_components=n_components)

    # 对节点表征进行降维
    reduced_features = pca.fit_transform(features)

    return reduced_features


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PCA Dimension Reduction')
    parser.add_argument('input_file', type=str, help='Input .npy file')
    parser.add_argument('n_components', type=int, help='Number of components for dimension reduction')
    args = parser.parse_args()

    # 读取输入文件
    input_file = args.input_file
    features = np.load(input_file)

    # 输出原始维度信息
    print("Original shape:", features.shape)

    # 进行降维操作
    reduced_features = reduce_dimension(features, n_components=args.n_components)

    # 保存降维后的结果到新文件
    output_file = input_file.replace('.npy', '_PCA.npy')
    np.save(output_file, reduced_features)

    # 输出降维后的维度信息
    print("Reduced shape:", reduced_features.shape)
    print("Saved to:", output_file)


if __name__ == '__main__':
    main()