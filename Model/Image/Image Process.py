import timm
from Model.Task.Node_Classification import ClassificationModel
# 加载预训练的图像特征提取器

class ImageClasifier(ClassificationModel):
    def __init__(self, config, model):
        super().__init__(config, model)

    def forward(self, x):


def get_image_model(feature_extractor, task, args):
    if task == 'node':
        model = timm.create_model(feature_extractor, pretrained=True, num_classes=args.num_classes)
    else:
        raise ValueError('Task is not supported')
    return model




