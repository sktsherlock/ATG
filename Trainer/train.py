import timm.optim
from Utils.utils import set_random_seed
from Model.Image.Image_Process import get_image_model
from Model.Text.Text_Process import get_text_model
from timm.data import create_dataset
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
import torch.nn as nn
from timm.data import ImageDataset
from timm.scheduler import CosineLRScheduler
import torch
import torchmetrics
from pytorch_accelerated.callbacks import SaveBestModelCallback
from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS


def get_model(attribute, feature_extractor, task, num_classes=None, device=None):
    if attribute == 'image':
        model = get_image_model(feature_extractor, task, num_classes=num_classes, device=device)
        return model
    elif attribute == 'text':
        model = get_text_model(feature_extractor, task)
        return model
    else:
        raise ValueError


def get_data(data_name, attribute, path, split='train'):
    if attribute == 'image':
        dataset = create_dataset(name='torch/cifar10', root=path + data_name, download=True, split=split)
    elif attribute == 'text':
        dataset = None
    else:
        raise ValueError
    return dataset, len(dataset.classes)


def create_model_and_optimizer(model):
    optimizer = timm.optim.AdamP(model.parameters(), lr=0.01)
    return model, optimizer


def create_dataloader_iterator(data_name, attribute, path=None, batch_size=2, split='train', shuffle=True):
    data, num_classes = get_data(data_name, attribute, path, split)
    print(data)
    dl = iter(DataLoader(data, batch_size=batch_size, shuffle=shuffle))
    return dl, num_classes


def run_exp(attribute, task, feature_extractor, data_name, args):
    # 根据属性,特征提取器,任务，构造模型并选择Trtainer
    for i in range(args.runs):
        # 种子随run变化而加一
        set_random_seed(args.seed + i)
        # 生成数据集与对应Dataloader

        train_dataloader, num_classes = create_dataloader_iterator(data_name, attribute, args.path, batch_size=2, split='train', shuffle=True)
        valid_dataloader, _ = create_dataloader_iterator(data_name, attribute, args.path, batch_size=2, split='validation', shuffle=False)

        loss_function = nn.CrossEntropyLoss()
        model = get_model(attribute, feature_extractor, task, num_classes, device=args.device)

        model, optimizer = create_model_and_optimizer(model)

        num_epoch_repeat = args.epochs / 2
        num_steps_per_epoch = 10
        scheduler = CosineLRScheduler(optimizer, t_initial=num_epoch_repeat * num_steps_per_epoch,
                                                     lr_min=1e-6, cycle_limit=num_epoch_repeat + 1, t_in_epochs=False,
                                                     warmup_lr_init=0.01, warmup_t=20)
        num_epochs = args.epochs + 10

        ema_model = ModelEmaV2(model, decay=0.9998)

        for epoch in range(num_epochs):
            for batch in train_dataloader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = loss_function(outputs, targets)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                ema_model.update(model)

            for batch in valid_dataloader:
                inputs, targets = batch
                outputs = model(inputs)
                validation_loss = loss_function(outputs, targets)

                ema_model_outputs = ema_model.module(inputs)
                ema_model_validation_loss = loss_function(ema_model_outputs, targets)
