from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import argparse
import torch
import os

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)


# processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--data_dir", type=str, default='Data/Movies/MoviesImages/', required=True,
                        help="The data folder on disk.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU to use")
    parser.add_argument('--model_name', type=str, default='resnet101d', help='Name or path of the CV model')
    parser.add_argument('--pretrained', type=bool, default=True, help='if load the pretrained weight')
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
    parser.add_argument('--path', type=str, default='Data/Movies/ImageFeature/', help='Path to save the NPY File')
    parser.add_argument("--size", type=int, default=224, help="The size of the image to load")
    parser.add_argument('--batch_size', type=int, default=100, help='Number of batch size for inference')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # Set training arguments, hardcoded here for clarity
    image_size = (args.size, args.size)
    batch_size = args.batch_size
    model_name = args.model_name
    name = args.name
    imagesize = args.size

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))
    data_dir = base_dir + '/' + args.data_dir

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state.shape)
