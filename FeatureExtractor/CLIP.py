from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch as th


def split_data(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


parser = argparse.ArgumentParser(
    description='Process text data and save the overall representation as an NPY file.')
parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                    help='Name or path of the Huggingface model')
parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
parser.add_argument('--csv_path', type=str, default='./', help='Where save the picture')
parser.add_argument('--path', type=str, default='./', help='Where save the picture')
parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
parser.add_argument('--feature_size', type=int, default=768, help='Number of batch size of CLIP image models')
args = parser.parse_args()

device = th.device("cuda" if th.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained(args.model_name).to(device)
processor = CLIPProcessor.from_pretrained(args.model_name)

root_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(root_dir.rstrip('/'))
picture_path = os.path.join(base_dir, args.path)

df = pd.read_csv(args.csv_path)
labels = df['label'].tolist()
categories = df['second_category'].unique().tolist()
num_classes = len(categories)
# print(categories)


# 获取文件夹中的所有图像文件
image_files = [filename for filename in os.listdir(picture_path) if filename.endswith((".jpg", ".png"))]
# 按照文件名的数字顺序排序
sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

clip_features = np.zeros((len(sorted_files), args.feature_size))  # 这里的feature_size是特征的维度
clip_probs = np.zeros((len(sorted_files), num_classes))
all_labels = []

train_ids, val_ids, test_ids = split_data(len(sorted_files), train_ratio=0.6, val_ratio=0.2)
print(train_ids, val_ids)
val_labels = np.array(labels)[val_ids]

for i, filename in enumerate(sorted_files):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(filename)
        image_path = os.path.join(picture_path, filename)
        image = Image.open(image_path)

        inputs = processor(text=[f"a {args.name} belonging to the '{category}'" for category in categories], images=image, return_tensors="pt", padding=True).to(device)
        print(f"a {args.name} belonging to the '{category}'" for category in categories)
        outputs = model(**inputs)
        feature = outputs.image_embeds

        # 将当前图像的特征添加到特征矩阵中
        clip_features[i] = feature.squeeze().detach().cpu().numpy()

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()
        print(probs)
        print('--------------------------------')
        clip_probs[i] = probs.squeeze()
        # 使用argmax获取预测的类别，并将其添加到类别列表中
        predicted_label = logits_per_image.argmax(dim=1).item()
        print(predicted_label, '---------------')
        all_labels.append(predicted_label)


print("已从文件夹中的所有图像中提取特征.")
# 将标签列表转换为NumPy数组并保存为npy文件
clip_labels = np.array(all_labels)
np.save('clip_labels.npy', clip_labels)

# 保存特征矩阵和概率矩阵为npy文件
np.save('clip_features.npy', clip_features)
np.save('clip_probs.npy', clip_probs)


# 计算准确率和F1指标
val_labels = labels[val_ids]
val_predictions = clip_labels[val_ids]

test_labels = labels[test_ids]
test_predictions = clip_labels[test_ids]


val_accuracy = accuracy_score(val_labels, val_predictions)
val_f1 = f1_score(val_labels, val_predictions)

test_accuracy = accuracy_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions)

# 打印结果
print("验证集上的准确率和F1指标：")
print("准确率：", val_accuracy)
print("F1指标：", val_f1)

print("测试集上的准确率和F1指标：")
print("准确率：", test_accuracy)
print("F1指标：", test_f1)