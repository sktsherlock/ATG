import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser(description='计算准确率和F1分数')
parser.add_argument('--clip_labels', type=str, default='all_labels.npy', help='CLIP模型的标签文件路径')
parser.add_argument('--csv_file', type=str, default='your_csv_file.csv', help='原始CSV文件路径')
args = parser.parse_args()

# 加载CLIP的标签和原始CSV文件
clip_labels = np.load(args.clip_labels)
df = pd.read_csv(args.csv_file)
true_labels = df['second_category'].tolist()
categories = df['second_category'].unique().tolist()

# 计算准确率和F1分数
accuracy = accuracy_score(true_labels, clip_labels)
f1 = f1_score(true_labels, clip_labels, average='macro')

# 打印结果
print("准确率: {:.2f}%".format(accuracy * 100))
print("F1 分数: {:.2f}".format(f1))