import torch.nn as nn
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput



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


class MEANClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask, labels):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        mean_emb = self.dropout(torch.mean(outputs.last_hidden_state, dim=1))
        logits = self.classifier(mean_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)


class AdapterClassifier(PreTrainedModel):
    def __init__(self, model, adapter, loss_func, dropout=0.0):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.adapter = adapter
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_ids, attention_mask, labels):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        mean_emb = self.dropout(torch.mean(outputs.last_hidden_state, dim=1))
        logits = self.adapter(mean_emb)
        # [conv(mean_emb) + mean_emb] -> classifier  -> logits
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)



class ResAdapterClassifier(PreTrainedModel):
    def __init__(self, model, graphTuner, n_labels, loss_func, dropout=0.0, scale=1.0):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.GraphTuner = graphTuner
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels)
        self.scale = scale



    def forward(self, input_ids, attention_mask, labels):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        mean_emb = self.dropout(torch.mean(outputs.last_hidden_state, dim=1))
        LLM_logits = self.classifier(mean_emb)
        GraphTune_logits = self.GraphTuner(mean_emb)
        logits = (1 - self.scale) * LLM_logits + self.scale * GraphTune_logits
        # [conv(mean_emb) + mean_emb] -> classifier  -> logits
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)