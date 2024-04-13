# yinjun@2024/04/13

import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.nn.pool import global_mean_pool, global_max_pool

# Command arguments
parser = argparse.ArgumentParser(description='PGExplainer')
# parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type=str, default='', help="Root directory where the dataset should be saved")
parser.add_argument("--batch_size", type=int, default=128, help="")
parser.add_argument("--device", type=str, default='cuda:0', help="")
parser.add_argument("--model_path", type=str, default='', help="Root directory where the trained model should be saved")
parser.add_argument("--hidden_dim", type=int, default=20, help="")
parser.add_argument("--lr", type=float, default=1e-3, help="")
parser.add_argument("--epochs", type=int, default=30, help="")
parser.add_argument("--eval_step", type=int, default=5, help="")
args = parser.parse_args()
print(args)

device = torch.device(args.device)

# Dataset
dataset = BA2MotifDataset(args.data_path)
'''
dataset: BA2MotifDataset(1000)
dataset[0]: Data(x=[25, 10], edge_index=[2, 50], y=[1])
dataset.num_features: 10
dataset.num_classes: 2
'''
# Dataloader
index = list(range(len(dataset)))
random.shuffle(index)
train_num = round(0.8 * len(dataset))
valid_num = round(0.9 * len(dataset))
trainset, validset, testset = dataset[index[:train_num]], dataset[index[train_num:valid_num]], dataset[
    index[valid_num:]]
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)


# GNN model
# Define a simple GCN.
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # (num_features, 20)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # (20, 20)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # (20, 20)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # (20, num_classes)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(data.x)
        print(data.edge_index)

        x = self.conv1(x, edge_index).relu()
        # (num_features -> 20)
        x = self.conv2(x, edge_index).relu()
        # (20 -> 20)
        x = self.conv3(x, edge_index).relu()
        # (20 -> 20)
        x = global_mean_pool(x, batch)
        # (node -> graph)
        x = self.linear(x)
        # (20 -> num_classes)

        return x


model = GCN(dataset.num_features, args.hidden_dim, dataset.num_classes)
model = model.to(device)
model.train()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()
# The input is expected to contain the unnormalized logits for each class
# (which do not need to be positive or sum to 1, in general).

best_accuracy = 0.0
# Train
for epoch in range(args.epochs):
    total_correct = []
    for data in tqdm(trainloader):
        data = data.to(device)
        # data.shape: (num_nodes, num_features)
        optimizer.zero_grad()
        output = model(data)
        # output.shape: (batchsize, num_classes)
        label = data.y
        # label.dim: (batchsize)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        # for k, v in model.named_parameters():
        #     print(v.grad)

        prediction = torch.argmax(output, dim=1)
        # prediction.shape: (batchsize)
        correct = prediction == label
        total_correct.extend(correct.detach().cpu().numpy().tolist())
    accuracy = np.sum(total_correct) / len(trainset)
    print('Epoch', epoch + 1, 'Accuracy:', accuracy)

    # Validation
    if (epoch + 1) % args.eval_step == 0:
        with torch.no_grad():
            val_total_correct = []
            for data in tqdm(validloader):
                data = data.to(device)
                output = model(data)
                label = data.y
                val_loss = loss_fn(output, label)

                prediction = torch.argmax(output, dim=1)
                val_correct = prediction == label
                val_total_correct.extend(val_correct.detach().cpu().numpy().tolist())
            val_accuracy = np.sum(val_total_correct) / len(validset)
            print('Epoch', epoch + 1, 'Validation Accuracy:', val_accuracy)

            if val_accuracy > best_accuracy:
                torch.save(model.state_dict(), args.model_path)

# Evaluate
eval_model = GCN(dataset.num_features, args.hidden_dim, dataset.num_classes)
eval_model.load_state_dict(args.model_path)
eval_model = eval_model.to(device)
eval_model.eval()
eval_total_correct = []
for data in tqdm(testloader):
    data = data.to(device)
    output = model(data)
    label = data.y
    prediction = torch.argmax(output, dim=1)
    eval_correct = prediction == label
    eval_total_correct.extend(eval_correct.detach().cpu().numpy().tolist())
eval_accuracy = np.sum(eval_total_correct) / len(testset)
print('Epoch', epoch + 1, 'Evaluation Accuracy:', eval_accuracy)