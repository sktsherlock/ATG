import argparse
import random
from pathlib import Path

import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torch
import torchmetrics
from pytorch_accelerated.callbacks import SaveBestModelCallback
from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS
from timm.scheduler import CosineLRScheduler


# _logger = logging.getLogger(__name__)
#
# _ERROR_RETRY = 50
#
#
# class ImageDataset(data.Dataset):
#
#     def __init__(
#             self,
#             root,
#             graph_path,
#             reader=None,
#             split='train',
#             class_map=None,
#             load_bytes=False,
#             input_img_mode='RGB',
#             transform=None,
#             target_transform=None,
#     ):
#         self.graph_path = graph_path
#         if reader is None or isinstance(reader, str):
#             reader = create_reader(
#                 reader or '',
#                 root=root,
#                 split=split,
#                 class_map=class_map
#             )
#         self.reader = reader
#         self.load_bytes = load_bytes
#         self.input_img_mode = input_img_mode
#         self.transform = transform
#         self.target_transform = target_transform
#         self._consecutive_errors = 0
#
#     def __getitem__(self, index):
#         graph = dgl.load_graphs(self.graph_path)[0]['0']
#         # labels = graph.ndata['label']
#         # neighbors = list(graph.adjacency_matrix().tolil().rows)
#
#         img, target = self.reader[index]
#
#         try:
#             img = img.read() if self.load_bytes else Image.open(img)
#         except Exception as e:
#             _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
#             self._consecutive_errors += 1
#             if self._consecutive_errors < _ERROR_RETRY:
#                 return self.__getitem__((index + 1) % len(self.reader))
#             else:
#                 raise e
#         self._consecutive_errors = 0
#
#         if self.input_img_mode and not self.load_bytes:
#             img = img.convert(self.input_img_mode)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if target is None:
#             target = -1
#         elif self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, neighbor_image
#
#     def __len__(self):
#         return len(self.reader)
#
#     def filename(self, index, basename=False, absolute=False):
#         return self.reader.filename(index, basename, absolute)
#
#     def filenames(self, basename=False, absolute=False):
#         return self.reader.filenames(basename, absolute)


def create_datasets(image_size, data_mean, data_std, train_path, val_path, test_path):
    train_transforms = timm.data.create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m7-mstd0.5-inc1",
    )

    eval_transforms = timm.data.create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )

    test_transforms = timm.data.create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )

    train_dataset = timm.data.dataset.ImageDataset(train_path, transform=train_transforms)
    eval_dataset = timm.data.dataset.ImageDataset(val_path, transform=eval_transforms)
    test_dataset = timm.data.dataset.ImageDataset(test_path, transform=test_transforms)

    return train_dataset, eval_dataset, test_dataset


class TimmMixupTrainer(Trainer):
    def __init__(self, eval_loss_fn, mixup_args, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = timm.data.Mixup(**mixup_args)

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.ema_accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.ema_model = None

    def create_scheduler(self):
        return timm.scheduler.CosineLRScheduler(
            self.optimizer,
            t_initial=self.run_config.num_epochs,
            cycle_decay=0.5,
            lr_min=1e-6,
            t_in_epochs=True,
            warmup_t=3,
            warmup_lr_init=1e-4,
            cycle_limit=1,
        )

    def training_run_start(self):
        # Model EMA requires the model without a DDP wrapper and before sync batchnorm conversion
        self.ema_model = timm.utils.ModelEmaV2(
            self._accelerator.unwrap_model(self.model), decay=0.9
        )
        if self.run_config.is_distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        print(xb, xb.shape)
        print(yb, yb.shape)
        mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

    def train_epoch_end(
            self,
    ):
        self.ema_model.update(self.model)
        self.ema_model.eval()

        if hasattr(self.optimizer, "sync_lookahead"):
            self.optimizer.sync_lookahead()

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            xb, yb = batch
            outputs = self.model(xb)
            val_loss = self.eval_loss_fn(outputs, yb)
            self.accuracy.update(outputs.argmax(-1), yb)

            ema_model_preds = self.ema_model.module(xb).argmax(-1)
            self.ema_accuracy.update(ema_model_preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()

        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.run_history.update_metric(
            "ema_model_accuracy", self.ema_accuracy.compute().cpu()
        )
        self.accuracy.reset()
        self.ema_accuracy.reset()


def main(data_path, model_name, image_size, lr, smoothing, mixup, cutmix, batch_size, bce_target_thresh, num_epochs):
    image_size = (image_size, image_size)
    random.seed(2024)

    data_path = Path(data_path)
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"
    num_classes = len(list(train_path.iterdir()))

    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes,
    )

    # Create model using timm
    model = timm.create_model(
        model_name=model_name, pretrained=False, num_classes=num_classes,
        drop_path_rate=0.05
    )

    # Load data config associated with the model to use in data augmentation pipeline
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config["mean"]
    data_std = data_config["std"]

    # Create training and validation datasets
    train_dataset, eval_dataset, test_dataset = create_datasets(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        image_size=image_size,
        data_mean=data_mean,
        data_std=data_std,
    )

    # Create optimizer
    optimizer = timm.optim.create_optimizer_v2(
        model, opt="lookahead_AdamW", lr=lr, weight_decay=0.01
    )

    # As we are using Mixup, we can use BCE during training and CE for evaluation
    train_loss_fn = timm.loss.BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    # Create trainer and start training
    trainer = TimmMixupTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=train_loss_fn,
        eval_loss_fn=validate_loss_fn,
        mixup_args=mixup_args,
        num_classes=num_classes,
        callbacks=[
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(watch_metric="accuracy", greater_is_better=True),
        ],
    )

    trainer.train(
        per_device_batch_size=batch_size,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        create_scheduler_fn=trainer.create_scheduler,
    )

    trainer.evaluate(dataset=test_dataset, per_device_batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--data_path", type=str, required=True, help="The data folder on disk.")
    parser.add_argument("--model_name", type=str, default="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
                        help="Model name.")
    parser.add_argument("--image_size", type=int, default=384, help="The width or the height of images.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--smoothing", type=float, default=0.1, help="")
    parser.add_argument("--mixup", type=float, default=0.2, help="")
    parser.add_argument("--cutmix", type=float, default=1.0, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--bce_target_thresh", type=float, default=0.2, help="")
    parser.add_argument("--num_epochs", type=int, default=40, help="The number of epochs.")
    args = parser.parse_args()
    main(args.data_path, args.model_name, args.image_size, args.lr, args.smoothing, args.mixup,
         args.cutmix, args.batch_size, args.bce_target_thresh, args.num_epochs)
