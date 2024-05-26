import argparse
import random
from pathlib import Path

import wandb
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


def create_datasets(image_size, data_mean, data_std, train_path, val_path, test_path):
    train_transforms = timm.data.create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m7-mstd0.5",
        # 结果为 RandAugment 的幅度为 7，mstd 为 0.5; m（整数）： rand 增强的大小;n（整数）：每个图像选择的变换操作的数量，这是可选的，默认设置为 2; mstd （浮点数）：所应用噪声幅度的标准偏差
    )
    # 上述为训练集中图像进行一些变化，但又不至于扰动太多；

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
    def __init__(self, eval_loss_fn, mixup_args, num_classes, warmup_lr_init, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = timm.data.Mixup(**mixup_args)

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.ema_accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, task='multiclass', average='macro')
        self.ema_f1 = torchmetrics.F1Score(num_classes=num_classes, task='multiclass', average='macro')
        self.warmup_lr_init = warmup_lr_init
        self.ema_model = None

    def create_scheduler(self):
        return timm.scheduler.CosineLRScheduler(
            self.optimizer,
            t_initial=self.run_config.num_epochs,
            cycle_decay=0.8,
            lr_min=1e-6,
            t_in_epochs=True,
            warmup_t=3,  # 也需要调整
            warmup_lr_init=self.warmup_lr_init,  # 需要根据lr来改动可能
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
            self.f1.update(outputs.argmax(-1), yb)

            ema_model_preds = self.ema_model.module(xb).argmax(-1)
            self.ema_accuracy.update(ema_model_preds, yb)
            self.ema_f1.update(ema_model_preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()

        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.run_history.update_metric(
            "ema_model_accuracy", self.ema_accuracy.compute().cpu()
        )
        self.run_history.update_metric("f1", self.f1.compute().cpu())
        self.run_history.update_metric(
            "ema_model_f1", self.ema_f1.compute().cpu()
        )
        wandb.log({f'Val_accuracy': self.accuracy.compute().cpu(), f'Val_f1': self.f1.compute().cpu()})
        self.accuracy.reset()
        self.f1.reset()
        self.ema_accuracy.reset()
        self.ema_f1.reset()


def main(cf):
    model_name = cf.model_name

    image_size = (cf.image_size, cf.image_size)
    random.seed(42)  # Need to change

    data_path = Path(cf.data_path)
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"
    num_classes = len(list(train_path.iterdir()))

    mixup_args = dict(
        mixup_alpha=cf.mixup,
        cutmix_alpha=cf.cutmix,
        label_smoothing=cf.smoothing,
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
        model, opt="lookahead_AdamW", lr=cf.lr, weight_decay=0.01
    )
    # Lookahead 是一种有趣的优化算法,它可以与其他常用的优化算法(如 SGD、Adam 等)结合使用,以提高模型训练的稳定性和收敛速度。

    # As we are using Mixup, we can use BCE during training and CE for evaluation
    train_loss_fn = timm.loss.BinaryCrossEntropy(
        target_threshold=cf.bce_target_thresh, smoothing=cf.smoothing
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
        warmup_lr_init=cf.warmup_lr_init,
        callbacks=[
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(watch_metric=f"{cf.metric}", greater_is_better=True),
        ],
    )
    # DEFAULT_CALLBACKS: 这是一组默认的回调函数,可能包括了诸如记录训练loss、验证loss等基础功能。
    # SaveBestModelCallback: 这个回调函数会在每个 epoch 结束时,根据 "accuracy" 指标判断当前模型是否是最佳模型,如果是则保存该模型。greater_is_better=True 表示准确率越高越好。

    trainer.train(
        per_device_batch_size=cf.batch_size,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=cf.num_epochs,
        create_scheduler_fn=trainer.create_scheduler,
    )

    test_metrics = trainer.evaluate(dataset=test_dataset, per_device_batch_size=cf.batch_size)
    wandb.log(test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--data_path", type=str, required=True, help="The data folder on disk.")
    parser.add_argument("--model_name", type=str, default="timm/vit_base_patch16_clip_224.openai",
                        help="Model name.")
    parser.add_argument("--metric", type=str, default="accuracy",
                        help="metric name.")
    parser.add_argument("--image_size", type=int, default=384, help="The width or the height of images.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--warmup_lr_init", type=float, default=1e-5, help="Initial warmup Learning rate.")
    parser.add_argument("--smoothing", type=float, default=0.1, help="")
    parser.add_argument("--mixup", type=float, default=0.2, help="")
    parser.add_argument("--cutmix", type=float, default=1.0, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--bce_target_thresh", type=float, default=0.2, help="")
    parser.add_argument("--num_epochs", type=int, default=40, help="The number of epochs.")
    arguments = parser.parse_args()
    wandb.init(config=arguments, reinit=True)

    main(arguments)
