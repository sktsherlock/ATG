import argparse
import os
import shutil

import numpy as np
import pandas as pd


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


def data_splitting(csv_path, photos_path, save_path, train_ratio=0.6, val_ratio=0.2):
    """
    :param csv_path: csv 路径
    :param photos_path: 源图片路径
    :param save_path: 目标图片路径
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """
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

    np.random.seed(2024)
    indices = np.random.permutation(node_nums)  # 打乱顺序

    train_size = int(node_nums * train_ratio)  # 训练集大小
    val_size = int(node_nums * val_ratio)  # 验证集大小
    test_size = node_nums - train_size - val_size  # 测试集大小

    train_idx = indices[:train_size]  # 训练集 id
    val_idx = indices[train_size:train_size + val_size]  # 验证集 id
    test_idx = indices[train_size + val_size:]  # 测试集 id

    train_labels = [hashmap[x] for x in train_idx]  # 训练集 label
    val_labels = [hashmap[x] for x in val_idx]  # 验证集 label
    test_labels = [hashmap[x] for x in test_idx]  # 测试集 label

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


def main(csv_path, photos_path, save_path, train_ratio=0.6, val_ratio=0.2):
    data_splitting(csv_path, photos_path, save_path, train_ratio=train_ratio, val_ratio=val_ratio
                   )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for dataset organization')
    parser.add_argument('--csv_path', type=str, help='Path to the output csv file')
    parser.add_argument('--photos_path', type=str, help='Path to the photos folder')
    parser.add_argument('--save_path', type=str, help='Path to the output photos folder')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    args = parser.parse_args()

    main(args.csv_path, args.photos_path, args.save_path, args.train_ratio, args.val_ratio)