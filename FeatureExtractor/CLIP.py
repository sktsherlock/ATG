from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as th

parser = argparse.ArgumentParser(
    description='Process text data and save the overall representation as an NPY file.')
parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                    help='Name or path of the Huggingface model')
parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
parser.add_argument('--csv_path', type=str, default='./', help='Where save the picture')
parser.add_argument('--path', type=str, default='./', help='Where save the picture')
parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
parser.add_argument('--feature_size', type=int, default=768, help='Number of batch size of CLIP image models')
args = parser.parse_args()

device = th.device("cuda" if th.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained(args.model_name).to(device)
processor = CLIPProcessor.from_pretrained(args.model_name)

root_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(root_dir.rstrip('/'))
picture_path = os.path.join(base_dir, args.path)

df = pd.read_csv(args.csv_path)
labels = df['second_category'].tolist()
categories = df['second_category'].unique().tolist()
num_classes = len(categories)
# print(categories)


# 获取文件夹中的所有图像文件
image_files = [filename for filename in os.listdir(picture_path) if filename.endswith((".jpg", ".png"))]
# 按照文件名的数字顺序排序
sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

all_features = np.zeros((len(sorted_files), args.feature_size))  # 这里的feature_size是特征的维度
all_probs =  np.zeros((len(sorted_files), num_classes))
all_labels = []

with tqdm(total=len(sorted_files)) as pbar:
    for i, filename in enumerate(sorted_files):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            image_path = os.path.join(picture_path, filename)
            image = Image.open(image_path)

            inputs = processor(text=[f"a {args.name} belonging to the '{category}'" for category in categories], images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            feature = outputs.image_embeds

            # 将当前图像的特征添加到特征矩阵中
            all_features[i] = feature.squeeze().detach().numpy()

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1).detach().numpy()
            all_probs[i] = probs.squeeze()
            # 使用argmax获取预测的类别，并将其添加到类别列表中
            predicted_label = logits_per_image.argmax(dim=1).item()
            all_labels.append(predicted_label)


print("已从文件夹中的所有图像中提取特征.")
# 将标签列表转换为NumPy数组并保存为npy文件
all_labels = np.array(all_labels)
np.save('all_labels.npy', all_labels)

# 保存特征矩阵和概率矩阵为npy文件
np.save('all_features.npy', all_features)
np.save('all_probs.npy', all_probs)