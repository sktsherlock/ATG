# Books 数据处理工具函数
import os
import re

import pandas as pd
import requests
import argparse
from data_processing_utils import  construct_graph, export_as_csv, parse_json, download_images

# 读取 json 文件并将其转换为 DataFrame 并返回
def parse_json_for_books(data_path):
    # 读取 json 文件
    df = pd.read_json(data_path, lines=True, orient='records')
    # 只取其中几列
    df = pd.DataFrame(df, columns=['asin', 'category', 'description', 'title',
                                   'also_buy', 'also_view', 'imageURL'])
    return df


# 数据过滤
def data_filter_for_books(df, category, category_numbers=10):
    # 过滤含有缺失数据和重复的记录
    df = df.drop_duplicates(subset=['asin'])
    df = df.dropna()
    # 删除 DataFrame 中含有空图像或不含三级类别(三级类别会作为分类的 label)的行
    mask = (df['imageURL'].str.len() >= 1) & (df['category'].str.len() >= 3)
    df = df[mask]
    print('步骤一****************************************************************')
    # 将 &amp; 替换为 &
    df['category'] = df['category'].apply(lambda x: [re.sub('&amp;', '&', i) for i in x])

    # 提取出第二类别
    df['second_category'] = df['category'].apply(lambda x: x[1] if x else None)

    # 只保留二级类别为指定类别的数据
    df['second_category'] = df['second_category'].apply(lambda x: x if category in x else None)
    df.dropna(subset=['second_category'], inplace=True)
    print('步骤二****************************************************************')
    # 提取出第三类别
    df['third_category'] = df['category'].apply(lambda x: x[2] if x else None)

    # 删除只有少数个数的 category
    category_counts = df['third_category'].value_counts()
    categories_to_keep = category_counts.nlargest(category_numbers).index

    df['third_category'] = df['third_category'].apply(lambda x: x if x in categories_to_keep else None)
    df.dropna(subset=['third_category'], inplace=True)
    print('步骤三****************************************************************')
    # 只保留 description 列表的第一项
    df['description'] = df['description'].apply(lambda x: x[0] if x else None)

    # 将其中以 '<' 开头的无效标题置为空
    df['title'] = df['title'].apply(lambda x: x if x and not re.match('^<', x) else None)
    print('步骤四****************************************************************')
    # 删除既未共同浏览又未共同购买的商品(即孤立商品)
    hash_set = set()
    existed = set()
    hash_table = {}  # 存储商品的 ASIN 到索引的映射, 便于删除操作
    # 初始化集合和字典
    for index, row in df.iterrows():
        hash_set.add(row['asin'])  # 将每个 ASIN 添加到集合, 若存在共同浏览或共同购买项则移除, 最终剩余的就是孤立商品
        existed.add(row['asin'])
        hash_table[row['asin']] = index

    for index, row in df.iterrows():
        # 移除含有共同购买项的商品
        if row['also_buy']:
            for asin in row['also_buy']:
                if asin in existed:
                    hash_set.discard(row['asin'])
                    hash_set.discard(asin)
        # 移除含有共同浏览项的商品
        if row['also_view']:
            for asin in row['also_view']:
                if asin in existed:
                    hash_set.discard(row['asin'])
                    hash_set.discard(asin)
    # 删除孤立商品
    rows_to_drop = []
    for asin in hash_set:
        rows_to_drop.append(hash_table[asin])
    df.drop(rows_to_drop, inplace=True)
    print('步骤五****************************************************************')
    # 清洗 description
    df['cleaned_description'] = df['description'].apply(
        lambda string: ''.join([c for c in string if c != '']) if string else None)
    # 替换 HTML 标签和空白
    df['cleaned_description'] = df['cleaned_description'].apply(lambda text: re.sub('<[\s\S]*>', '',
                                                                                    re.sub('\s+', ' ',
                                                                                           text)) if text else None)
    # 若还有 '<', 则删除
    df['cleaned_description'] = df['cleaned_description'].apply(lambda x: x if x and not re.search('<', x) else None)
    print('步骤六****************************************************************')
    # 合并 description 和 title
    df['text'] = df.apply(
        lambda per_row: 'Description: {}; Title: {}'.format(per_row['cleaned_description'], per_row['title']), axis=1)
    # 再次替换 HTML 标签和空白
    df['text'] = df['text'].apply(lambda text: re.sub('<[\s\S]*>', '',
                                                      re.sub('\s+', ' ', text)) if text else None)
    # 将 &amp 替换为 &
    df['text'] = df['text'].apply(lambda x: re.sub('&amp;', '&', x))
    print('步骤七****************************************************************')

    # 将商品的 ASIN 映射为递增的 id 以便后续数据处理
    df = df.reset_index(drop=True)  # 重置索引
    hash_table = {}
    for index, row in df.iterrows():
        hash_table[row['asin']] = int(index)
    df['asin'] = df['asin'].map(hash_table)  # 将 ASIN 映射为 id
    df.rename(columns={'asin': 'id'}, inplace=True)  # 将列名 asin 更改为 id
    print('步骤八****************************************************************')
    # 将 also_buy 与 also_view 中的商品 ASIN 索引替换为 id 索引, 同时剔除不存在项
    df['also_buy'] = df['also_buy'].apply(lambda x: [hash_table[i] for i in x if i in hash_table])
    df['also_view'] = df['also_view'].apply(lambda x: [hash_table[i] for i in x if i in hash_table])
    print('步骤九****************************************************************')
    # 将三级类别映射为递增的 label
    hash_table = {}
    label_number = 1
    for index, row in df.iterrows():
        if row['third_category'] not in hash_table:
            hash_table[row['third_category']] = label_number
            label_number += 1
    df['label'] = df['third_category'].map(hash_table)  # 三级类别映射为递增的 label

    # 只保留 DataFrame 中需要的列
    df = df[['id', 'category', 'text', 'also_buy', 'also_view', 'imageURL', 'third_category', 'label']]

    return df


# 爬取 DataFrame 中的图片并保存
def download_images_for_books(df, output_img_path):
    print('Downloading images...')
    total = len(df)
    for index, row in df.iterrows():
        if row['imageURL']:
            need_deleted = True  # 是否需要删除该商品
            for image_url in row['imageURL']:
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
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    parser.add_argument('--name', type=str, help='Dataset short name parameter', required=True)
    parser.add_argument('--class_numbers', type=int, help='Dataset class threshold', required=True)
    parser.add_argument('--second_category', type=str, default='Computer')
    args = parser.parse_args()

    data_path = args.data_path
    name = args.name
    class_numbers = args.class_numbers
    second_category = args.second_category

    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')

    output_csv_path = f'./{name}/{name}.csv'
    output_img_path = f'./{name}/{name}Images'
    output_graph_path = f'./{name}/{name}Graph.pt'

    df = data_filter_for_books(parse_json_for_books(data_path), args.second_category,  category_numbers=class_numbers)
    export_as_csv(df, output_csv_path)
    construct_graph(output_csv_path, output_graph_path)
    # 从本地读取处理后的CSV文件
    download_images(df, output_img_path)
