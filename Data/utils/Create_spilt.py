import argparse
import os
import shutil
import dgl
import numpy as np
import pandas as pd


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

    return train_size, val_size, train_ids, val_ids, test_ids


def change_path(csv_path, photos_path, save_path):
    """
    更改图片路径
    :param csv_path: csv 路径
    :param photos_path: 源图片路径
    :param save_path: 目标图片路径
    :return: None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_csv(csv_path)
    category_numbers = df['label'].nunique()  # 类别数

    for i in range(category_numbers):
        class_path = os.path.join(save_path, str(i))
        if not os.path.exists(class_path):
            os.makedirs(class_path)  # 创建文件夹

    for index, row in df.iterrows():
        src_path = str(os.path.join(photos_path, '{}.jpg'.format(row['id'])))
        dest_path = str(os.path.join(save_path, str(row['label']), '{}.jpg'.format(row['id'])))
        shutil.copy(src_path, dest_path)


def data_splitting(csv_path, photos_path, save_path, gpth, train_ratio=0.6, val_ratio=0.2):
    change_path(csv_path, photos_path, save_path)  # 更改图片路径

    train_folder = os.path.join(save_path, 'train')  # 训练集路径
    val_folder = os.path.join(save_path, 'val')  # 验证集路径
    test_folder = os.path.join(save_path, 'test')  # 测试集路径
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    category_numbers = df['label'].nunique()  # 类别数
    node_nums = len(df)  # 节点数
    hashmap = {}
    for index, row in df.iterrows():
        hashmap[row['id']] = row['label']

    graph = dgl.load_graphs(gpth)[0][0]
    labels = graph.ndata['label']

    train_size, val_size, train_idx, val_idx, test_idx = split_graph(nodes_num=node_nums, train_ratio=train_ratio, val_ratio=val_ratio, labels=labels, fewshots=None)
    test_size = node_nums - train_size - val_size

    train_labels = [hashmap[x] for x in train_idx]  # 训练集 label
    val_labels = [hashmap[x] for x in val_idx]  # 验证集 label
    test_labels = [hashmap[x] for x in test_idx]  # 测试集 label
    print(train_labels, val_labels, test_labels)

    for i in range(category_numbers):
        os.makedirs(os.path.join(save_path, 'train', str(i)), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'val', str(i)), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test', str(i)), exist_ok=True)

    for index in range(train_size):  # 更改训练集
        src_file = os.path.join(save_path, str(train_labels[index]), '{}.jpg'.format(train_idx[index]))
        dest_file = os.path.join(train_folder, str(train_labels[index]), '{}.jpg'.format(train_idx[index]))
        shutil.move(src_file, dest_file)  # 移动图片

    for index in range(val_size):  # 更改验证集
        src_file = os.path.join(save_path, str(val_labels[index]), '{}.jpg'.format(val_idx[index]))
        dest_file = os.path.join(val_folder, str(val_labels[index]), '{}.jpg'.format(val_idx[index]))
        shutil.move(src_file, dest_file)  # 移动图片

    for index in range(test_size):  # 更改测试集
        src_file = os.path.join(save_path, str(test_labels[index]), '{}.jpg'.format(test_idx[index]))
        dest_file = os.path.join(test_folder, str(test_labels[index]), '{}.jpg'.format(test_idx[index]))
        shutil.move(src_file, dest_file)  # 移动图片

    for i in range(category_numbers):
        os.rmdir(os.path.join(save_path, str(i)))


def main(csv_path, photos_path, save_path, graph_path, train_ratio=0.6, val_ratio=0.2):
    data_splitting(csv_path, photos_path, save_path, graph_path, train_ratio=train_ratio, val_ratio=val_ratio
                   )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for dataset organization')
    parser.add_argument('--csv_path', type=str, help='Path to the output csv file', required=True)
    parser.add_argument('--graph_path', type=str, help='Path to the graph path', required=True)
    parser.add_argument('--photos_path', type=str, help='Path to the photos folder', required=True)
    parser.add_argument('--save_path', type=str, help='Path to the output photos folder', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    args = parser.parse_args()

    main(args.csv_path, args.photos_path, args.save_path, args.graph_path, args.train_ratio, args.val_ratio)