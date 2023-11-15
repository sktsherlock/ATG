import timm

def get_text_model(feature_extractor, task, args):
    if task == 'node':
        model = timm.create_model(feature_extractor, pretrained=True, num_classes=args.num_classes)
    else:
        raise ValueError('Task is not supported')
    return model




