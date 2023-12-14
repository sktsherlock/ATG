import pandas as pd
import argparse


def count_words(csv_file, column_name):
    # 读取 CSV 文件为 DataFrame
    data = pd.read_csv(csv_file)

    # 统计每行的单词数
    data['word_count'] = data[column_name].str.split().str.len()
    data['text_length'] = data.apply(lambda x: len(x[f'{column_name}'].split(' ')) if x[f'{column_name}'] else 0, axis=1)

    # 输出结果
    print(data['word_count'].describe())
    print('***************************')
    print(data['text_length'].describe())




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', required=True)
    parser.add_argument('--column_name', type=str, help='The column for the text', required=True)
    args = parser.parse_args()
    count_words(args.csv_file, args.column_name)
