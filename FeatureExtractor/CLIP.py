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
parser.add_argument('--feature_path', type=str, default='./', help='Where to save the feature')
args = parser.parse_args()

device = th.device("cuda" if th.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained(args.model_name).to(device)
processor = CLIPProcessor.from_pretrained(args.model_name)

# root_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.dirname(root_dir.rstrip('/'))
picture_path = args.path

df = pd.read_csv(args.csv_path)
labels = df['label'].tolist()
if args.name in {'RedditS', 'Reddit'}:
    categories = df['subreddit'].unique().tolist()
else:
    categories = df['second_category'].unique().tolist()
num_classes = len(categories)

# 获取文本描述
image_texts = df['text'].tolist()  # 假设文本列名为'text'

if not os.path.exists(args.feature_path):
    os.makedirs(args.feature_path)

# 获取文件夹中的所有图像文件
image_files = [filename for filename in os.listdir(picture_path) if filename.endswith((".jpg", ".png"))]
# 按照文件名的数字顺序排序
sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

clip_image_features = np.zeros((len(sorted_files), args.feature_size))
clip_text_features = np.zeros((len(sorted_files), args.feature_size))
clip_probs = np.zeros((len(sorted_files), num_classes))
all_labels = []

train_ids, val_ids, test_ids = split_data(len(sorted_files), train_ratio=0.6, val_ratio=0.2)
output_image_feature = f'{args.feature_path}/{args.name}_openai_clip-vit-large-patch14_image.npy'
output_text_feature = f'{args.feature_path}/{args.name}_openai_clip-vit-large-patch14_text.npy'
output_probs = f'{args.feature_path}/{args.name}_clip_probs.npy'
output_labels = f'{args.feature_path}/{args.name}_clip_labels.npy'
print(f'The output files are {output_image_feature} and {output_text_feature}')

if not os.path.exists(output_image_feature):
    for i, filename in tqdm(enumerate(sorted_files), total=len(sorted_files)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(picture_path, filename)
            image = Image.open(image_path)
            text = image_texts[i]  # 获取对应的文本描述

            # 处理图像
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_outputs = model.get_image_features(**image_inputs)
            clip_image_features[i] = image_outputs.squeeze().detach().cpu().numpy()

            # 处理文本
            max_length = model.config.text_config.max_position_embeddings
            text_inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            text_outputs = model.get_text_features(**text_inputs)
            clip_text_features[i] = text_outputs.squeeze().detach().cpu().numpy()

            # 计算类别概率
            category_inputs = processor(text=[f"a {args.name} belonging to the '{category}'" for category in categories], return_tensors="pt", padding=True).to(device)
            category_outputs = model.get_text_features(**category_inputs)
            logits_per_image = image_outputs @ category_outputs.T
            probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()
            clip_probs[i] = probs.squeeze()

            predicted_label = logits_per_image.argmax(dim=1).item()
            all_labels.append(predicted_label)

    print("已从文件夹中的所有图像和文本中提取特征.")
    np.save(output_image_feature, clip_image_features)
    np.save(output_text_feature, clip_text_features)
    np.save(output_probs, clip_probs)

    clip_labels = np.array(all_labels)
    np.save(output_labels, clip_labels)
else:
    print('Existing features, please load!')
    clip_image_features = np.load(output_image_feature)
    clip_text_features = np.load(output_text_feature)
    clip_labels = np.load(output_labels)




# 计算准确率和F1指标
val_labels = np.array(labels)[val_ids]
val_predictions = clip_labels[val_ids]

test_labels = np.array(labels)[test_ids]
test_predictions = clip_labels[test_ids]


val_accuracy = accuracy_score(val_labels, val_predictions)
val_macro_f1 = f1_score(val_labels, val_predictions,  average='macro')

test_accuracy = accuracy_score(test_labels, test_predictions)
test_macro_f1 = f1_score(test_labels, test_predictions, average='macro')


# 打印结果
print("验证集上的准确率和F1指标：")
print("准确率：", val_accuracy)
print("macro F1指标：", val_macro_f1)


print("测试集上的准确率和F1指标：")
print("准确率：", test_accuracy)
print("macro F1指标：", test_macro_f1)