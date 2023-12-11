from transformers import AutoModel
from peft import PeftModelForFeatureExtraction, get_peft_config

config = {
    "peft_type": "LORA",
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 16,
    "target_modules": [],
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "fan_in_fan_out": False,
    "bias": "none",
}
peft_config = get_peft_config(config)
model = AutoModel.from_pretrained("bert-base-cased")
peft_model = PeftModelForFeatureExtraction(model, peft_config)
peft_model.print_trainable_parameters()
print(peft_model)




