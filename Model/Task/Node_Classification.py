import torch.nn as nn


class NodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class ClassificationModel(nn.Module):

    def __init__(self, config, model):
        super().__init__()

        self.config = config

        self.n_hidden = self.config.d_model
        self.out = self.config.num_classes

        self.embedding = model

        self.classifier = NodeClassifier(self.n_hidden, self.n_hidden, self.out)

    def forward(self, x):
        x = self.embedding(x=x)
        a = self.classifier(x)
        return a
