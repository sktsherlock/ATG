import argparse
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration, Qwen2VLForConditionalGeneration, \
    PaliGemmaForConditionalGeneration
from tqdm import tqdm


class PaliGemmaFeatureExtractor:
    def __init__(self, model_name, device):
        self.device = device
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract_features(self, image, text):
        # Determine the number of images
        num_images = 1  # if isinstance(image, (list, tuple)) else len(image)

        # Add <image> tokens at the beginning of the text
        image_tokens = "<image>" * num_images
        updated_text = image_tokens + text
        # messages = [
        #     {"role": "user", "content": [
        #         {"type": "image", "source": image},
        #         {"type": "text", "text": text}
        #     ]}
        # ]
        # input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            text=updated_text,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]

        TV_features = last_hidden_state.mean(dim=1)
        TV_features = TV_features.float()

        return TV_features.cpu().numpy()

    def extract_text_features(self, text):
        inputs = self.processor(
            text=text,
            images=None,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]

        text_features = last_hidden_state.mean(dim=1)
        text_features = text_features.float()

        return text_features.cpu().numpy()

    def extract_image_features(self, image):
        num_images = 1  # if isinstance(image, (list, tuple)) else len(image)

        # Add <image> tokens at the beginning of the text
        image_tokens = "<image>" * num_images
        inputs = self.processor(
            text=image_tokens,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # 获取最后一层隐藏状态
        last_hidden_state = outputs.hidden_states[-1]

        # 计算图像特征
        image_features = last_hidden_state.mean(dim=1)
        image_features = image_features.float()

        return image_features.cpu().numpy()


class MultimodalLLaMAFeatureExtractor:
    def __init__(self, model_name, device):
        self.device = device
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract_features(self, image, text):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": image},
                {"type": "text", "text": text}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]

        TV_features = last_hidden_state.mean(dim=1)
        TV_features = TV_features.float()

        return TV_features.cpu().numpy()

    def extract_image_features(self, image):

        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": image}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # # 打印最后一层隐藏状态的形状
            # print(f"Extracted image features shape: {outputs.hidden_states[-1].shape}")

        # 获取最后一层隐藏状态
        last_hidden_state = outputs.hidden_states[-1]

        # 计算图像特征
        image_features = last_hidden_state.mean(dim=1)
        image_features = image_features.float()

        return image_features.cpu().numpy()


class QWenFeatureExtractor:
    def __init__(self, model_name, device):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).to(self.device)
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)

    def extract_features(self, image, text):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": image},
                {"type": "text", "text": text}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]

        TV_features = last_hidden_state.mean(dim=1)
        TV_features = TV_features.float()

        return TV_features.cpu().numpy()

    def extract_image_features(self, image):

        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": image}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # # 打印最后一层隐藏状态的形状
            print(f"Extracted image features shape: {outputs.hidden_states[-1].shape}")

        # 获取最后一层隐藏状态
        last_hidden_state = outputs.hidden_states[-1]

        # 计算图像特征
        image_features = last_hidden_state.mean(dim=1)
        image_features = image_features.float()

        return image_features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Process text and image data and save the overall representation as NPY files.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct',
                        help='Name or path of the Huggingface MLLM model')
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the NPY file')
    parser.add_argument('--csv_path', type=str, default='./', help='Path to the CSV file')
    parser.add_argument('--image_path', type=str, default='./', help='Path to the image directory')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
    parser.add_argument('--path', type=str, default='./', help='Where to save the features')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to process for testing')
    parser.add_argument('--text_column', type=str, default='text',
                        help='The name of the column containing the text data')
    parser.add_argument('--feature_type', type=str, default='tv', choices=['text', 'visual', 'tv'],
                        help='Type of features to extract (text, image, or multimodal)')
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))
    Feature_path = os.path.join(base_dir, args.path)
    print(Feature_path)

    if not os.path.exists(Feature_path):
        os.makedirs(Feature_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'llama' in args.model_name.lower():
        extractor = MultimodalLLaMAFeatureExtractor(args.model_name, device)
    elif 'qwen' in args.model_name.lower():
        extractor = QWenFeatureExtractor(args.model_name, device)
    elif 'paligemma' in args.model_name.lower():
        extractor = PaliGemmaFeatureExtractor(args.model_name, device)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    picture_path = args.image_path
    df = pd.read_csv(args.csv_path)

    image_texts = df[args.text_column].tolist()

    image_files = [filename for filename in os.listdir(picture_path) if filename.endswith((".jpg", ".png"))]
    sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    # If sample_size is provided, use it; otherwise, use the full dataset
    if args.sample_size:
        sorted_files = sorted_files[:args.sample_size]
        image_texts = image_texts[:args.sample_size]

    output_features = np.zeros((len(sorted_files),))

    args.model_name = args.model_name.split('/')[-1]
    output_feature = f'{Feature_path}/{args.name}_{args.model_name}_{args.feature_type}.npy'

    print(f'The output files are {output_feature}')

    if not os.path.exists(output_feature):
        if args.feature_type == 'text':
            for i, text in tqdm(enumerate(image_texts), total=len(image_texts)):
                text_feature = extractor.extract_text_features(text)

                if i == 0:
                    output_features = np.zeros((len(image_texts), text_feature.shape[1]))

                output_features[i] = text_feature
        elif args.feature_type == 'visual':
            for i, filename in tqdm(enumerate(sorted_files), total=len(sorted_files)):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(picture_path, filename)
                    image = Image.open(image_path).convert("RGB")
                    visual_feature = extractor.extract_image_features(image)

                    if i == 0:
                        output_features = np.zeros((len(sorted_files), visual_feature.shape[1]))

                    output_features[i] = visual_feature
        elif args.feature_type == 'tv':
            for i, filename in tqdm(enumerate(sorted_files), total=len(sorted_files)):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(picture_path, filename)
                    image = Image.open(image_path).convert("RGB")
                    text = image_texts[i]
                    tv_feature = extractor.extract_features(image, text)

                    if i == 0:
                        output_features = np.zeros((len(sorted_files), tv_feature.shape[1]))

                    output_features[i] = tv_feature

        print("Features extracted from all images and texts.")
        np.save(output_feature, output_features)
    else:
        print('Existing features, please load!')
        output_features = np.load(output_feature)

    print("Multimodal TV features shape:", output_features.shape)


if __name__ == "__main__":
    main()
