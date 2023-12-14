import numpy as np
import pandas as pd
import argparse
from ogb.utils.url import download_url
from ogb.nodeproppred import DglNodePropPredDataset
import os
import re


def remove_special_characters(text):
    # 使用正则表达式或其他方法去除特殊字符
    cleaned_text = re.sub('[^a-zA-Z0-9]', '', text)
    return cleaned_text


def remove_html_tags(text):
    # 去除 HTML 标签
    cleaned_text = re.sub('<[^>]+>', '', text)
    return cleaned_text


def remove_whitespace(text):
    # 去除多余空白字符
    cleaned_text = ' '.join(text.split())
    return cleaned_text


def clean_df(data, title='title', abstract='abstract'):
    # 清洗 'title' 列
    data[title] = data[title].apply(remove_special_characters)
    data[title] = data[title].apply(remove_html_tags)
    data[title] = data[title].apply(remove_whitespace)

    # 清洗 'abstract' 列
    data[abstract] = data[abstract].apply(remove_special_characters)
    data[abstract] = data[abstract].apply(remove_html_tags)
    data[abstract] = data[abstract].apply(remove_whitespace)
    return data




def merge_by_ids(meta_data, node_ids, categories):
    meta_data.columns = ["ID", "Title", "Abstract"]
    # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full dataset processing
    meta_data["ID"] = meta_data["ID"].astype(np.int64)
    meta_data.columns = ["mag_id", "title", "abstract"]
    data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
    data = pd.merge(data, categories, how="left", on="label_id")
    return data


def read_ids_and_labels(data_root):
    category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
    paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  #
    paper_ids = pd.read_csv(paper_id_path_csv)
    categories = pd.read_csv(category_path_csv)
    categories.columns = ["ID", "category"]  # 指定ID 和 category列写进去
    paper_ids.columns = ["ID", "mag_id"]
    categories.columns = ["label_id", "category"]

    return categories, paper_ids  # 返回类别和论文ID


def process_raw_text_df(meta_data, node_ids, categories, process_mode='TA'):
    data = merge_by_ids(meta_data.dropna(), node_ids, categories)
    data = clean_df(data)

    return data


# Get Raw text path
def main(raw_url, data_path):
    raw_text_path = download_url(raw_url, data_path)
    categories, node_ids = read_ids_and_labels(data_path)
    text = pd.read_table(raw_text_path, header=None, skiprows=[0])
    arxiv_csv = process_raw_text_df(text, node_ids, categories)
    # 保存ogb-arxiv文件
    arxiv_csv.to_csv(output_csv_path, sep=',', index=False, header=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Path to the data file', required=True)
    args = parser.parse_args()

    if not os.path.exists(f'./ogb'):
        os.makedirs(f'./ogb')

    data_root = args.data_root
    output_csv_path = f'./ogb/arxiv.csv'
    raw_text_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"

    dataset = DglNodePropPredDataset('ogbn-arxiv', root=data_root)

    main(raw_url=raw_text_url, data_path=os.path.join(data_root, 'ogbn_arxiv'))
