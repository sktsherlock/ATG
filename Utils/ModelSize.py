from transformers import AutoModel
import argparse


def print_trainable_parameters(encoder):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in encoder.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for check the size of the model')
    parser.add_argument('--model_name', type=str, help='Path to the model or the name',
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    args = parser.parse_args()

    # 创建用于分类的模型
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        token='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',
    )
    print_trainable_parameters(model)

