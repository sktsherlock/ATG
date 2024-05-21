import torch.nn as nn
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


def mean_pooling(token_embeddings, attention_mask):
    # Mask out padding tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    masked_token_embeddings = token_embeddings * input_mask_expanded
    # Calculate mean pooling
    mean_embeddings = masked_token_embeddings.sum(dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return mean_embeddings


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

    def forward(self, input_ids, attention_mask, labels, nb_input_ids=None, nb_attention_mask=None):
        # Extract outputs from the model
        if nb_input_ids is not None:
            topology_ids = torch.cat([input_ids, nb_input_ids], dim=1)
            topology_attention_mask = torch.cat([attention_mask, nb_attention_mask], dim=1)
            outputs = self.encoder(topology_ids, topology_attention_mask, output_hidden_states=True)
            mean_emb = self.dropout(mean_pooling(outputs.last_hidden_state, topology_attention_mask))
        else:
            outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
            mean_emb = self.dropout(mean_pooling(outputs.last_hidden_state, attention_mask))
            # mean_emb = self.dropout(torch.mean(outputs.last_hidden_state, dim=1))
        logits = self.classifier(mean_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)



class DualClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, inputs_dim, dropout=0.0):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.alignment = nn.Linear(inputs_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask, labels, nb_input_ids=None, nb_attention_mask=None, knowledge_embedding=None):
        # Extract outputs from the model
        if nb_input_ids is not None:
            topology_ids = torch.cat([input_ids, nb_input_ids], dim=1)
            topology_attention_mask = torch.cat([attention_mask, nb_attention_mask], dim=1)
            text_representation = self.encoder.embeddings(topology_ids)

            mean_emb = self.dropout(mean_pooling(outputs.last_hidden_state, topology_attention_mask))
        else:

            outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
            mean_emb = self.dropout(mean_pooling(outputs.last_hidden_state, attention_mask))
            # mean_emb = self.dropout(torch.mean(outputs.last_hidden_state, dim=1))
        logits = self.classifier(mean_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)




class TopologicalCL(PreTrainedModel):
    def __init__(self, PLM, dropout=0.0, projection_dim=128):
        super().__init__(PLM.config)
        self.dropout = nn.Dropout(dropout)
        hidden_dim = PLM.config.hidden_size
        self.text_encoder = PLM

        self.project = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim))

    def forward(self, input_ids=None, attention_mask=None, nb_input_ids=None, nb_attention_mask=None):
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        center_node_emb = self.dropout(mean_pooling(center_node_outputs.last_hidden_state, attention_mask))

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, output_hidden_states=True
        )

        toplogy_emb = self.dropout(mean_pooling(toplogy_node_outputs.last_hidden_state, attention_mask))

        center_contrast_embeddings = self.project(center_node_emb)
        toplogy_contrast_embeddings = self.project(toplogy_emb)

        return center_contrast_embeddings, toplogy_contrast_embeddings
