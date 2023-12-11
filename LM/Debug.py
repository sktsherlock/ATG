from transformers import AutoModel
from peft import PeftModelForFeatureExtraction, get_peft_config

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

config = {
    "peft_type": "LORA",
    "inference_mode": False,
    "r": 16,
    "target_modules": ["query", "key"],
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "fan_in_fan_out": False,
    "bias": "none",
}
peft_config = get_peft_config(config)
model = AutoModel.from_pretrained("bert-base-cased")
print_trainable_parameters(model)
peft_model = PeftModelForFeatureExtraction(model, peft_config)
peft_model.print_trainable_parameters()
print(peft_model)




