from transformers import AutoModelForSeq2SeqLM, AutoModel
from peft import LoraModel, LoraConfig


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v", "k"],
    lora_dropout=0.01,
)

model = AutoModel.from_pretrained('roberta-large')
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
print_trainable_parameters(model)
print([(n, type(m)) for n, m in model.named_modules()])
lora_model = LoraModel(model, config, "default")
print_trainable_parameters(lora_model)
print(lora_model)
