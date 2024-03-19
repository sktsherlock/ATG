import numpy as np
import pandas as pd
import argparse
from ogb.utils.url import download_url
from ogb.nodeproppred import DglNodePropPredDataset
import os
import re





def merge_by_ids(meta_data, node_ids, categories):
    meta_data.columns = ["ID", "Title", "Abstract"]
    # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full dataset processing
    meta_data["ID"] = meta_data["ID"].astype(np.int64)
    meta_data.columns = ["mag_id", "title", "abstract"]
    data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
    data = pd.concat([data, label_df], axis=1)
    data = pd.merge(data, categories, how="left", on="label_id")
    return data


def read_ids_and_labels(data_root):
    category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
    paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  #
    paper_ids = pd.read_csv(paper_id_path_csv)
    categories = pd.read_csv(category_path_csv)
    paper_ids.columns = ["ID", "mag_id"]
    categories.columns = ["label_id", "category"]

    return categories, paper_ids  # 返回类别和论文ID


def process_raw_text_df(meta_data, node_ids, categories):
    data = merge_by_ids(meta_data.dropna(), node_ids, categories)
    data['title'] = data.apply(lambda per_row: 'Title: {}'.format(per_row['title']), axis=1)
    # data['title'] = data.apply(lambda per_row: '{}'.format(per_row['title']), axis=1)
    data['abstract'] = data.apply(lambda per_row: 'Abstract: {}'.format(per_row['abstract']), axis=1)
    # data['abstract'] = data.apply(lambda x: ' '.join(x['abstract'].split(' ')[:args.max_length]), axis=1)
    # print(data['abstract'])
    # data['prompt_category'] = data.apply(lambda per_row: 'This paper belongs to the {} sub-category of arXiv Computer Science (cs) field.'.format(per_row['category']), axis=1)
    # Merge title and abstract
    data['text'] = data.apply(
        lambda per_row: '{}. {}'.format(per_row['title'], per_row['abstract']), axis=1)
    return data


# Get Raw text path
def main(raw_url, data_path):
    raw_text_path = download_url(raw_url, data_path)
    categories, node_ids = read_ids_and_labels(data_path)
    text = pd.read_table(raw_text_path, header=None, skiprows=[0])
    arxiv_csv = process_raw_text_df(text, node_ids, categories)
    # 保存ogb-arxiv文件
    if args.save:
        arxiv_csv.to_csv(output_csv_path, sep=',', index=False, header=True)

    # 统计每行的单词数
    arxiv_csv['word_count'] = arxiv_csv[column_name].str.split().str.len()
    arxiv_csv['text_length'] = arxiv_csv.apply(lambda x: len(x[f'{column_name}'].split(' ')) if x[f'{column_name}'] else 0, axis=1)

    # 输出结果
    print(arxiv_csv['word_count'].describe())
    print('***************************')
    print(arxiv_csv['text_length'].describe())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/aiscuser/TAG', type=str, help='Path to the data file')
    parser.add_argument('--save', default=False, type=bool,
                        help='Whether to save the csv file')
    # parser.add_argument('--max_length', type=int, default=1024, help='Few shot')
    parser.add_argument('--column_name', type=str, default="text", help='The column for the text')
    args = parser.parse_args()

    column_name = args.column_name
    data_root = args.data_root
    output_csv_path = f'/home/aiscuser/TAG/Arxiv/OGBN_ARXIV.csv'
    raw_text_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"

    dataset = DglNodePropPredDataset('ogbn-arxiv', root=data_root)
    _, label = dataset[0]
    label_array = label.numpy().flatten()
    label_df = pd.DataFrame({'label_id': label_array})

    main(raw_url=raw_text_url, data_path=os.path.join(data_root, 'ogbn_arxiv'))
