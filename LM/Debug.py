from transformers import AutoModel, AutoConfig
from peft import PeftModelForFeatureExtraction, get_peft_config
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn


class CLSClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask, labels):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        cls_emb = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)


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



def set_peft_config(config):
    config.peft_type = "LORA"
    config.target_modules = ["query", "key"]
    config.r = 16
    config.bias = "none"
    config.lora_alpha = 32
    config.lora_dropout = 0.05
    # config.layers_to_transform = modeling_args.lora_layers_to_transform
    # config = {'peft_type': modeling_args.peft_type, 'target_modules': modeling_args.lora_target_modules,
    #           'r': modeling_args.lora_rank, 'bias': modeling_args.lora_train_bias,
    #           'lora_alpha': modeling_args.lora_alpha, 'lora_dropout': modeling_args.lora_dropout,
    #           'layers_to_transform': modeling_args.lora_layers_to_transform}
    # peft_config = get_peft_config(config)
    return config







config = {
    "peft_type": "LORA",
    "r": 16,
    "target_modules": ["query", "key"],
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
}

peft_config = get_peft_config(config)

encoder = AutoModel.from_pretrained("bert-base-cased")
print_trainable_parameters(encoder)
peft_model = PeftModelForFeatureExtraction(encoder, peft_config)
peft_model.print_trainable_parameters()
print(peft_model)

model = CLSClassifier(
    peft_model, 10,
    loss_func=nn.CrossEntropyLoss(reduction='mean')
)

print(model)
print_trainable_parameters(model)

