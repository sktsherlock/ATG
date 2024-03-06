import argparse
import os
import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torch
import numpy as np
from timm.data.loader import create_loader
from transformers.trainer import nested_concat, DistributedTensorGatherer, nested_numpify
from typing import List, Union


def create_datasets(image_size, data_mean, data_std, inference_path):
    inference_transforms = timm.data.create_transform(
        input_size=image_size,
        mean=data_mean,
        std=data_std,
    )

    inference_dataset = timm.data.dataset.ImageDataset(
        root=inference_path, transform=inference_transforms
    )

    return inference_dataset

"""
python ImageExtract.py --gpu 1 --data_dir /home/aiscuser/ATG/Data/Movies/MoviesImages/ --name Movies --path /home/aiscuser/ATG/Data/Movies/ImageFeature/ 
"""
def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--data_dir", type=str, default='Data/Movies/MoviesImages/', required=True, help="The data folder on disk.")
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
    print(data_dir)


    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # Create model using timm
    model = timm.create_model(
        model_name, pretrained=args.pretrained, num_classes=0).to(device)

    # Load data config associated with the model to use in data augmentation pipeline
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config["mean"]
    data_std = data_config["std"]

    # Create training and validation datasets
    dataset = create_datasets(
        inference_path=data_dir,
        image_size=image_size,
        data_mean=data_mean,
        data_std=data_std,
    )

    dl = create_loader(dataset, (3, image_size[0], image_size[1]), batch_size, device=device)

    preds_gatherer = DistributedTensorGatherer(1, len(dataset))

    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None

    model.eval()

    for batch in dl:
        with torch.no_grad():
            inputs, _ = batch
            logits = model(inputs)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

                # Gather all tensors and put them back on the CPU
                preds_gatherer.add_arrays(nested_numpify(preds_host))
                preds_host = None

    preds = preds_gatherer.finalize()
    print(f'Shape of the image feature{preds.shape}')

    output_file = base_dir + '/' + args.path + name + '_' + model_name.split('/')[-1].replace("-", "_") + '_' + str(imagesize) + '_' + str(preds.shape[1])
    print(f'Base directory: {base_dir}')
    print(f'Output file: {output_file}')
    np.save(output_file + ".npy", preds)


if __name__ == "__main__":
    main()