# 数据处理工具函数
import json
import os
import time
import pandas as pd
import requests
import warnings
import argparse
import torch
import dgl
import ast
from tqdm import tqdm

# 忽略特定警告
warnings.filterwarnings("ignore",
                        message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")


# 读取 json 文件并将其转换为 DataFrame 并返回
def parse_json(data_path):
    # 读取 json 文件
    with open(data_path) as json_file:
        data = json.load(json_file)

    data = data['annotations']

    if data and all(key in data[0] for key in ['image_id', 'subreddit', 'url', 'caption', 'author']):
        df = pd.DataFrame(data)
        df = df[['image_id', 'subreddit', 'url', 'caption', 'author']]
        return df
    else:
        return None


def count_data(df):
    print('Counting data category...')
    # 统计类别信息
    print(df['subreddit'].value_counts())
    print(df['label'].value_counts())

    # 统计文本单词长度信息
    print('Counting data text length...')
    df2 = pd.DataFrame(df)
    df2['text_length'] = df.apply(lambda x: len(x['caption'].split(' ')) if x['caption'] else 0, axis=1)
    print(df2['text_length'].describe())


# 数据过滤
def data_filter_for_reddit(df, category_number=50, sampling=15000):
    # 过滤含有缺失数据和重复的记录
    df = df.drop_duplicates(subset=['image_id'])
    df = df.dropna()

    df = df[df['author'] != 'None']  # 去除作者为空的行
    df = df.reset_index(drop=True)  # 重置索引

    # 删除 url为空的行
    df.dropna(subset=['url'], inplace=True)

    # 选择类别数最多的几个帖子
    subreddit_counts = df['subreddit'].value_counts()
    subreddit_to_keep = subreddit_counts.nlargest(category_number).index
    print(f'The large subreddit are: {subreddit_to_keep}')
    df['subreddit'] = df['subreddit'].apply(lambda x: x if x in subreddit_to_keep else None)
    df.dropna(subset=['subreddit'], inplace=True)

    # 定义函数来对每个subreddit进行采样
    def subreddit_sampling(group):
        return group.sample(n=sampling, random_state=42)

    # 对每个subreddit进行采样
    new_df = df.groupby('subreddit', group_keys=False).apply(subreddit_sampling)
    new_df.drop_duplicates(inplace=True)
    # 重置索引并删除多余的列
    new_df.reset_index(drop=True, inplace=True)





    hash_set = {}
    for index, row in new_df.iterrows():
        if row['author'] not in hash_set:
            hash_set[row['author']] = [row['image_id']]
        else:
            hash_set[row['author']].append(row['image_id'])

    new_df['also_posted'] = None

    for index, row in new_df.iterrows():
        new_df.at[index, 'also_posted'] = hash_set[row['author']][:]

    # 删除孤立帖子
    new_df = new_df[new_df['also_posted'].str.len() >= 2]
    new_df = new_df.reset_index(drop=True)  # 重置索引


    # 将帖子的 image_id 映射为递增的 id 以便后续数据处理
    hash_table = {}
    for index, row in new_df.iterrows():
        hash_table[row['image_id']] = int(index)
    new_df['id'] = new_df['image_id'].map(hash_table)  # 将 image_id 映射为 id

    # 将 also_posted 中的帖子 image_id 索引替换为 id 索引, 同时剔除不存在项
    new_df['also_posted'] = new_df['also_posted'].apply(lambda x: [hash_table[i] for i in x if i in hash_table])

    # 将类别映射为递增的 label
    hash_table = {}
    label_number = 0
    for index, row in new_df.iterrows():
        if row['subreddit'] not in hash_table:
            hash_table[row['subreddit']] = label_number
            label_number += 1
    new_df['label'] = new_df['subreddit'].map(hash_table)  # 类别映射为递增的 label

    # 只保留 DataFrame 中需要的列
    new_df = new_df[['id', 'subreddit', 'caption', 'url', 'also_posted', 'label']]


    return new_df


def construct_graph(input_csv_path, output_graph):
    df = pd.read_csv(input_csv_path)

    df['also_posted'] = df['also_posted'].apply(lambda x: ast.literal_eval(x))

    df['neighbour'] = df['also_posted']
    df['neighbour'] = df['neighbour'].apply(lambda x: list(set(x)))
    df['neighbour'] = df['neighbour'].apply(lambda x: sorted(x))

    adj_list = {}
    asin_ids = sorted(df['id'].unique())
    for asin_id in asin_ids:
        adj_list[asin_id] = []
    for index, row in df.iterrows():
        src_asin_id = row['id']
        neighbour_asins = row['neighbour']
        for dest_asin in neighbour_asins:
            adj_list[src_asin_id].append(dest_asin)
    adj0 = []
    adj1 = []
    for src_asin_id, neighbors in adj_list.items():
        for dest_asin_id in neighbors:
            adj0.append(src_asin_id)
            adj1.append(dest_asin_id)
    # %%
    adj0 = torch.tensor(adj0)
    adj1 = torch.tensor(adj1)
    # %%
    G = dgl.graph((adj0, adj1))
    G.ndata['label'] = torch.tensor(list(df['label']))
    dgl.save_graphs(f"{output_graph}", G)
    print(G)


# 将 DataFrame 导出为 csv 文件
def export_as_csv(df, output_csv_path):
    df.to_csv(output_csv_path, sep=',', index=False, header=True)
    print('Successfully exported as CSV')


# 爬取 DataFrame 中的图片并保存
def download_images(df, output_img_path):
    print('Downloading images...')
    total = len(df)

    downloaded_images = set()  # 已下载的图像编号集合

    # 检查已下载的图像文件
    for filename in os.listdir(output_img_path):
        if filename.endswith('.jpg'):
            image_id = int(filename.split('.')[0])
            downloaded_images.add(image_id)

    for index, row in df.iterrows():
        need_deleted = True  # 是否需要删除该商品
        # for image_url in row['url']:
        image_name = '{}.jpg'.format(int(index))  # 图像命名为 '商品id.jpg'
        image_path = os.path.join(output_img_path, image_name)

        if index in downloaded_images:  # 图像已经下载过，跳过当前循环
            need_deleted = False
            if (index + 1) % 1000 == 0:
                print('Downloaded {} items\' images, {} in total'.format(index + 1, total))
            continue

        if not os.path.exists(output_img_path):
            os.makedirs(output_img_path)
        image_data = requests.get(row['url']).content  # 获取图像数据


        if not image_data.lower() == 'Not Found'.encode('utf-8').lower():  # 图像存在
            need_deleted = False  # 不需要删除该商品
            if (index + 1) % 1000 == 0:
                print('Downloaded {} items\' images, {} in total'.format(index + 1, total))
            with open(image_path, 'wb') as f:
                f.write(image_data)
            continue
        if need_deleted:
            print('No.{} need to be deleted'.format(int(index)))
    print('Successfully downloaded images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--name', type=str, help='Dataset short name parameter', required=True)
    parser.add_argument('--class_numbers', type=int, help='Dataset class numbers', required=True)
    parser.add_argument('--sampling', type=int, help='Dataset class numbers', default=15000)
    parser.add_argument('--download_image', action='store_true', help='whether to download the image')
    parser.add_argument('--save', action="store_true", help="is saving or not")
    args = parser.parse_args()

    name = args.name
    class_numbers = args.class_numbers

    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')

    initial_csv_path = f'./{name}/{name}_raw.csv'
    output_csv_path = f'./{name}/{name}.csv'
    output_img = f'./{name}/{name}Images'
    output_graph_path = f'./{name}/{name}Graph.pt'

    # Reddit/annotations/
    folder_path = args.data_path
    # 获取文件夹内所有文件名
    file_names = os.listdir(folder_path)

    if output_csv_path is not None:
        data = pd.read_csv(output_csv_path)
        # data['url'] = data['url'].apply(lambda x: ast.literal_eval(x))
        if args.download_image:
            download_images(data, output_img)
    else:
        data = pd.DataFrame(None, columns=['image_id', 'subreddit', 'url', 'caption', 'author'])
        if initial_csv_path is not None:
            data = pd.read_csv(initial_csv_path)
        else:
            for file_name in tqdm(file_names, desc='Processing Files'):
                df = parse_json(os.path.join(folder_path, file_name))
                if df is not None:
                    data = data.append(df)

            data.to_csv(initial_csv_path, sep=',', index=False, header=True)

        # 记录代码开始执行的时间
        start_time = time.time()
        data = data_filter_for_reddit(data, class_numbers, args.sampling)
        # 记录代码执行结束的时间
        end_time = time.time()
        # 计算代码执行的时间
        execution_time = end_time - start_time
        # 打印代码执行时间
        print("代码执行时间：", execution_time, "秒")
        count_data(data)

        if args.save is True:
            export_as_csv(data, output_csv_path)
            construct_graph(output_csv_path, output_graph_path)
            # 从本地读取处理后的CSV文件
            if args.download_image:
                download_images(data, output_img)
        else:
            print('Check Finished.')
        # df = data_filter_for_reddit(df)
        # print(df)
