import timm

def get_image_model(feature_extractor, task, num_classes=10, device=None):
    if task == 'node':
        model = timm.create_model(feature_extractor, pretrained=False, num_classes=num_classes).to(device)
    else:
        raise ValueError('Task is not supported')
    return model




