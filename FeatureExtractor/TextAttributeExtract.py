import argparse
import numpy as np
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModel, TrainingArguments, PreTrainedModel, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


# 定义命令行参数
parser = argparse.ArgumentParser(description='Process text data and save the overall representation as an NPY file.')
parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny', help='Name or path of the Huggingface model')
parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
parser.add_argument('--fp16', type=bool, default=False, help='if fp16')

# 解析命令行参数
args = parser.parse_args()
csv_file = args.csv_file
text_column = args.text_column
model_name = args.model_name
name = args.name
max_length = args.max_length
batch_size = args.batch_size
inf_path = f"{args.path}cache/"

if not os.path.exists(args.path):
    os.makedirs(args.path)
if not os.path.exists(inf_path):
    os.makedirs(inf_path)

output_file = args.path + name + '_' + model_name.split('/')[-1].replace("-", "_") + '_' + str(max_length)

class CLSEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.encoder = model
    @torch.no_grad()
    def forward(self, inputs):
        # Extract outputs from the model
        outputs = self.encoder(**inputs)
        # Use CLS Emb as sentence emb.
        node_cls_emb = outputs.last_hidden_state[:, 0, :]  # Last layer
        return TokenClassifierOutput(logits=node_cls_emb)

class MeanEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.encoder = model
    @torch.no_grad()
    def forward(self, inputs):
        # Extract outputs from the model
        outputs = self.encoder(**inputs)
        # Use Mean Emb as sentence emb.
        node_mean_emb = torch.mean(outputs.last_hidden_state, dim=1)
        return TokenClassifierOutput(logits=node_mean_emb)
# 读取CSV文件
df = pd.read_csv(csv_file)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 处理文本数据并进行推理
text_data = df[text_column].tolist()

# 编码文本数据
encoded_inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt', max_length=max_length)





dataset = Dataset.from_dict(encoded_inputs)


model = AutoModel.from_pretrained(model_name)

CLS_Feateres_Extractor = CLSEmbInfModel(model)
Mean_Features_Extractor = MeanEmbInfModel(model)
CLS_Feateres_Extractor.eval()
Mean_Features_Extractor.eval()

inference_args = TrainingArguments(
    output_dir=inf_path,
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=batch_size,
    dataloader_drop_last=False,
    dataloader_num_workers=1,
    fp16_full_eval=False,
)

trainer = Trainer(model=CLS_Feateres_Extractor, args=inference_args)
out_cls_emb, out_mean_emb = trainer.predict(dataset)

with torch.no_grad():
    output = model(**encoded_inputs)

# 提取CLS表示
cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
# 提取平均表示
mean_embeddings = torch.mean(output.last_hidden_state, dim=1).cpu()
# 保存整体表示为NPY文件
np.save(output_file + "_cls.npy", cls_embeddings)
np.save(output_file + "_mean.npy", mean_embeddings)