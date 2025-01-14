import pandas as pd
import argparse


def count_words(csv_file, column_name, threshold, target_row):
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

    count_greater_than_threshold = len(data[data['text_length'] > threshold])
    total_count = len(data)
    percentage = (count_greater_than_threshold / total_count) * 100

    print(f"行数大于{threshold}的记录数: {count_greater_than_threshold}")
    print(f"总记录数: {total_count}")
    print(f"占比: {percentage}%")

    # 显示指定行及其上下行的文本长度
    if target_row is not None:
        start_row = max(0, target_row - 1)
        end_row = min(len(data), target_row + 2)
        print(f"\nText length for rows {start_row} to {end_row - 1}:")
        for i in range(start_row, end_row):
            print(f"Row {i}: {data['text_length'].iloc[i]}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', required=True)
    parser.add_argument('--column_name', type=str, help='The column for the text', required=True)
    parser.add_argument('--threshold', type=int, help='The column for the text', default=512)
    parser.add_argument('--target_row', type=int, help='The row number to display text length for')
    args = parser.parse_args()
    count_words(args.csv_file, args.column_name, args.threshold, args.target_row)
