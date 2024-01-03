import pandas as pd
import argparse


def count_words(csv_file, column_name):
    # 读取 CSV 文件为 DataFrame
    data = pd.read_csv(csv_file)
    column_names = data.columns.tolist()
    print(column_names)
    # 输出前后几行
    num_rows = 5  # 指定要输出的行数
    head_data = data.head(num_rows)
    print(head_data)
    tail_data = data.tail(num_rows)
    print(tail_data)

    if column_name == 'Debug':
        return
    # 统计每行的单词数
    data['word_count'] = data[column_name].str.split().str.len()
    data['text_length'] = data.apply(lambda x: len(x[f'{column_name}'].split(' ')) if x[f'{column_name}'] else 0, axis=1)

    # 输出结果
    print(data['word_count'].describe())
    print('***************************')
    print(data['text_length'].describe())

    max_length = data['text_length'].max()
    max_length_row = data[data['text_length'] == max_length]
    print(f'The max length row is in :{max_length_row}')
    # max_length_row_text = max_length_row[column_name].iloc[0]
    # print(max_length_row_text)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', required=True)
    parser.add_argument('--column_name', type=str, help='The column for the text', default='Debug')
    args = parser.parse_args()
    count_words(args.csv_file, args.column_name)
