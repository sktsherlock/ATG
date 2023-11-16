import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# 定义命令行参数
parser = argparse.ArgumentParser(description='Process text data and save the overall representation as an NPY file.')
parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny', help='Name or path of the Huggingface model')
parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解析命令行参数
args = parser.parse_args()
csv_file = args.csv_file
text_column = args.text_column
model_name = args.model_name
name = args.name
max_length = args.max_length

output_file = args.path + name + '_' + model_name.split('/')[-1].replace("-", "_") + '_' + str(max_length)

# 读取CSV文件
df = pd.read_csv(csv_file)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# 处理文本数据并进行推理
text_data = df[text_column].tolist()

# 编码文本数据
encoded_inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
with torch.no_grad():
    output = model(**encoded_inputs)

# 提取CLS表示
cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
# 提取平均表示
mean_embeddings = torch.mean(output.last_hidden_state, dim=1).cpu()
# 保存整体表示为NPY文件
np.save(output_file + "_cls.npy", cls_embeddings)
np.save(output_file + "_mean.npy", mean_embeddings)