from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='Process text data and save the overall representation as an NPY file.')
parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                    help='Name or path of the Huggingface model')
parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
parser.add_argument('--csv_path', type=str, default='./', help='Where save the picture')
parser.add_argument('--path', type=str, default='./', help='Where save the picture')
parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
args = parser.parse_args()

model = CLIPModel.from_pretrained(args.model_name)
processor = CLIPProcessor.from_pretrained(args.model_name)

root_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(root_dir.rstrip('/'))
picture_path = os.path.join(base_dir, args.path)

df = pd.read_csv(args.csv_path)
labels = df['second_category'].tolist()
categories = df['second_category'].unique().tolist()
print(categories)




for filename in os.listdir(picture_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print(filename)
        image_path = os.path.join(picture_path, filename)
        image = Image.open(image_path)

        inputs = processor(text=[f"a {args.name} belonging to the '{category}'" for category in categories], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        feature = outputs.image_embeds
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities




        # 在这里处理每个图像的特征，可以将它们保存到一个列表或其他数据结构中

print("已从文件夹中的所有图像中提取特征.")


