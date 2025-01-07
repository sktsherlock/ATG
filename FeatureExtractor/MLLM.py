import argparse
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
from tqdm import tqdm


class MultimodalLLaMAFeatureExtractor:
    def __init__(self, model_name, device):
        self.device = device
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
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
        inputs = self.processor(messages=messages, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        text_features = last_hidden_state.mean(dim=1)
        image_features = self.model.vision_model(inputs.pixel_values).last_hidden_state.mean(dim=1)

        return text_features.cpu().numpy(), image_features.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(
        description='Process text and image data and save the overall representation as NPY files.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct',
                        help='Name or path of the Huggingface model')
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the NPY file')
    parser.add_argument('--csv_path', type=str, default='./', help='Path to the CSV file')
    parser.add_argument('--path', type=str, default='./', help='Path to the image directory')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
    parser.add_argument('--feature_size', type=int, default=768, help='Size of the feature vectors')
    parser.add_argument('--feature_path', type=str, default='./', help='Where to save the features')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = MultimodalLLaMAFeatureExtractor(args.model_name, device)

    picture_path = args.path
    df = pd.read_csv(args.csv_path)
    labels = df['label'].tolist()

    if args.name in {'RedditS', 'Reddit'}:
        categories = df['subreddit'].unique().tolist()
    else:
        categories = df['second_category'].unique().tolist()
    num_classes = len(categories)

    image_texts = df['text'].tolist()

    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path)

    image_files = [filename for filename in os.listdir(picture_path) if filename.endswith((".jpg", ".png"))]
    sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    llama_text_features = np.zeros((len(sorted_files), args.feature_size))
    llama_image_features = np.zeros((len(sorted_files), args.feature_size))

    output_text_feature = f'{args.feature_path}/{args.name}_llama_text.npy'
    output_image_feature = f'{args.feature_path}/{args.name}_llama_image.npy'
    print(f'The output files are {output_text_feature} and {output_image_feature}')

    if not os.path.exists(output_text_feature):
        for i, filename in tqdm(enumerate(sorted_files), total=len(sorted_files)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(picture_path, filename)
                image = Image.open(image_path).convert("RGB")
                text = image_texts[i]

                text_feature, image_feature = extractor.extract_features(image, text)
                llama_text_features[i] = text_feature
                llama_image_features[i] = image_feature

        print("Features extracted from all images and texts.")
        np.save(output_text_feature, llama_text_features)
        np.save(output_image_feature, llama_image_features)
    else:
        print('Existing features, please load!')
        llama_text_features = np.load(output_text_feature)
        llama_image_features = np.load(output_image_feature)

    print("Text features shape:", llama_text_features.shape)
    print("Image features shape:", llama_image_features.shape)

if __name__ == "__main__":
    main()
