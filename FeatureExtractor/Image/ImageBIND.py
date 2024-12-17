import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
from PIL import Image
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def split_data(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


parser = argparse.ArgumentParser(description='Process image and text data using ImageBIND model.')
parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the NPY file')
parser.add_argument('--csv_path', type=str, default='./', help='Path to the CSV file')
parser.add_argument('--path', type=str, default='./', help='Path to save the pictures')
parser.add_argument('--feature_size', type=int, default=1024, help='Size of the feature vector')
parser.add_argument('--feature_path', type=str, default='./', help='Where to save the features')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载ImageBIND模型
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

df = pd.read_csv(args.csv_path)
labels = df['label'].tolist()
if args.name in {'RedditS', 'Reddit'}:
    categories = df['subreddit'].unique().tolist()
else:
    categories = df['second_category'].unique().tolist()
num_classes = len(categories)

# 获取文本描述
image_texts = df['text'].tolist()

if not os.path.exists(args.feature_path):
    os.makedirs(args.feature_path)

# 获取文件夹中的所有图像文件
image_files = [filename for filename in os.listdir(args.path) if filename.endswith((".jpg", ".png"))]
# 按照文件名的数字顺序排序
sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

imagebind_image_features = np.zeros((len(sorted_files), args.feature_size))
imagebind_text_features = np.zeros((len(sorted_files), args.feature_size))
imagebind_probs = np.zeros((len(sorted_files), num_classes))
all_labels = []

train_ids, val_ids, test_ids = split_data(len(sorted_files), train_ratio=0.6, val_ratio=0.2)
output_image_feature = f'{args.feature_path}/{args.name}_imagebind_image.npy'
output_text_feature = f'{args.feature_path}/{args.name}_imagebind_text.npy'
output_probs = f'{args.feature_path}/{args.name}_imagebind_probs.npy'
output_labels = f'{args.feature_path}/{args.name}_imagebind_labels.npy'
print(f'The output files are {output_image_feature} and {output_text_feature}')

if not os.path.exists(output_image_feature):
    for i, filename in tqdm(enumerate(sorted_files), total=len(sorted_files)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(args.path, filename)
            text = image_texts[i]

            # 准备输入
            inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data([image_path], device),
                ModalityType.TEXT: data.load_and_transform_text([text], device),
            }

            # 提取特征
            with torch.no_grad():
                embeddings = model(inputs)

            imagebind_image_features[i] = embeddings[ModalityType.VISION].squeeze().cpu().numpy()
            imagebind_text_features[i] = embeddings[ModalityType.TEXT].squeeze().cpu().numpy()

            # 计算类别概率
            category_inputs = {
                ModalityType.TEXT: data.load_and_transform_text(
                    [f"a {args.name} belonging to the '{category}'" for category in categories], device)
            }
            with torch.no_grad():
                category_embeddings = model(category_inputs)

            logits_per_image = embeddings[ModalityType.VISION] @ category_embeddings[ModalityType.TEXT].T
            probs = torch.softmax(logits_per_image, dim=1).cpu().numpy()
            imagebind_probs[i] = probs.squeeze()

            predicted_label = logits_per_image.argmax(dim=1).item()
            all_labels.append(predicted_label)

    print("已从文件夹中的所有图像和文本中提取特征.")
    np.save(output_image_feature, imagebind_image_features)
    np.save(output_text_feature, imagebind_text_features)
    np.save(output_probs, imagebind_probs)

    imagebind_labels = np.array(all_labels)
    np.save(output_labels, imagebind_labels)
else:
    print('Existing features, please load!')
    imagebind_image_features = np.load(output_image_feature)
    imagebind_text_features = np.load(output_text_feature)
    imagebind_labels = np.load(output_labels)

# 计算准确率和F1指标
val_labels = np.array(labels)[val_ids]
val_predictions = imagebind_labels[val_ids]

test_labels = np.array(labels)[test_ids]
test_predictions = imagebind_labels[test_ids]

val_accuracy = accuracy_score(val_labels, val_predictions)
val_macro_f1 = f1_score(val_labels, val_predictions, average='macro')

test_accuracy = accuracy_score(test_labels, test_predictions)
test_macro_f1 = f1_score(test_labels, test_predictions, average='macro')

# 打印结果
print("验证集上的准确率和F1指标：")
print("准确率：", val_accuracy)
print("macro F1指标：", val_macro_f1)

print("测试集上的准确率和F1指标：")
print("准确率：", test_accuracy)
print("macro F1指标：", test_macro_f1)
