# 数据处理工具函数
import json
import os
import time
import pandas as pd
import requests
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")


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


# 数据过滤
def data_filter_for_reddit(df):
    # 过滤含有缺失数据和重复的记录
    df = df.drop_duplicates(subset=['image_id'])
    df = df.dropna()

    df = df[df['author'] != 'None']  # 去除作者为空的行
    df = df.reset_index(drop=True)  # 重置索引

    hash_set = {}
    for index, row in df.iterrows():
        if row['author'] not in hash_set:
            hash_set[row['author']] = [row['image_id']]
        else:
            hash_set[row['author']].append(row['image_id'])

    df['also_posted'] = None

    for index, row in df.iterrows():
        df.at[index, 'also_posted'] = hash_set[row['author']][:]

    # 删除孤立帖子
    df = df[df['also_posted'].str.len() >= 2]
    df = df.reset_index(drop=True)  # 重置索引

    # 将帖子的 image_id 映射为递增的 id 以便后续数据处理
    hash_table = {}
    for index, row in df.iterrows():
        hash_table[row['image_id']] = int(index)
    df['id'] = df['image_id'].map(hash_table)  # 将 image_id 映射为 id

    # 将 also_posted 中的帖子 image_id 索引替换为 id 索引, 同时剔除不存在项
    df['also_posted'] = df['also_posted'].apply(lambda x: [hash_table[i] for i in x if i in hash_table])

    # 将类别映射为递增的 label
    hash_table = {}
    label_number = 0
    for index, row in df.iterrows():
        if row['subreddit'] not in hash_table:
            hash_table[row['subreddit']] = label_number
            label_number += 1
    df['label'] = df['subreddit'].map(hash_table)  # 类别映射为递增的 label

    # 只保留 DataFrame 中需要的列
    df = df[['id', 'subreddit', 'caption', 'also_posted', 'label']]

    return df


# 将 DataFrame 导出为 csv 文件
def export_as_csv(df, output_csv_path):
    df.to_csv(output_csv_path, sep=',', index=False, header=True)
    print('Successfully exported as CSV')


# 爬取 DataFrame 中的图片并保存
def download_images(df, output_img_path):
    print('Downloading images...')
    total = len(df)
    for index, row in df.iterrows():
        if row['imageURLHighRes']:
            need_deleted = True  # 是否需要删除该商品
            for image_url in row['imageURLHighRes']:
                image_name = '{}.jpg'.format(int(index))  # 图像命名为 '商品id.jpg'
                image_path = os.path.join(output_img_path, image_name)
                if not os.path.exists(output_img_path):
                    os.makedirs(output_img_path)
                image_data = requests.get(image_url).content  # 获取图像数据

                if not image_data.lower() == 'Not Found'.encode('utf-8').lower():  # 图像存在
                    need_deleted = False  # 不需要删除该商品
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    break
            if need_deleted:
                print('No.{} need to be deleted'.format(int(index)))
        if (index + 1) % 50 == 0:
            print('Downloaded {} items\' images, {} in total'.format(index + 1, total))
    print('Successfully downloaded images')


# df = parse_json('Reddit/annotations/abandoned_2017.json')

folder_path = 'Reddit/annotations/'
# 获取文件夹内所有文件名
file_names = os.listdir(folder_path)
data = pd.DataFrame(None, columns=['image_id', 'subreddit', 'url', 'caption', 'author'])
for file_name in file_names:
    print(file_name)
    df = parse_json(os.path.join(folder_path, file_name))
    if df is not None:
        data = data.append(df)

# 记录代码开始执行的时间
start_time = time.time()
data = data_filter_for_reddit(data)
# 记录代码执行结束的时间
end_time = time.time()

# 计算代码执行的时间
execution_time = end_time - start_time

# 打印代码执行时间
print("代码执行时间：", execution_time, "秒")

print(data)

# df = data_filter_for_reddit(df)
# print(df)
