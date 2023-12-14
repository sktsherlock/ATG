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
    data['abstract'] = data.apply(lambda per_row: 'Abstract: {}'.format(per_row['abstract']), axis=1)
    data['category'] = data.apply(lambda per_row: 'This paper belongs to the {} sub-category of arXiv Computer Science (cs) field.'.format(per_row['category']), axis=1)
    # Merge title and abstract
    data['text'] = data.apply(
        lambda per_row: '{} {}'.format(per_row['title'], per_row['abstract']), axis=1)
    data['TC'] = data.apply(
        lambda per_row: '{} {}'.format(per_row['title'], per_row['abstract']), axis=1)
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
    _, label = dataset[0]
    label_array = label.numpy().flatten()
    label_df = pd.DataFrame({'label_id': label_array})

    main(raw_url=raw_text_url, data_path=os.path.join(data_root, 'ogbn_arxiv'))
