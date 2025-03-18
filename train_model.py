"""
This script trains a model on WikiCS for a number of epochs with the full data
This shows a single model retraining from the node dropping experiment
"""
import argparse
import pickle
import time
import warnings

import numpy as np
import torch
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, WikiCS
import torch_geometric as pyg

from node_drop_large import SGCNet, get_subgraph_data
from pc_winter_run import calculate_md5_of_string, set_masks_from_indices, dataset_params, standardize_features


class SGC(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        # when no training nodes available initialize weights randomly
        weights = pyg.nn.dense.linear.reset_weight_(
            torch.empty(out_channels, in_channels), in_channels
        )
        self.weights = nn.Parameter(weights.T)

    def fit(self, x, edge_index, y, train_mask):
        device = x.device
        adj = pyg.utils.to_dense_adj(edge_index, max_num_nodes=x.shape[0])[0].to(device)
        if adj.diagonal().sum() != x.shape[0]:
            adj = adj + torch.eye(x.shape[0], device=device)  # self-loops

        deg = adj.sum(1)  # same for bidirectional and self-connected graphs
        deg_inv_sqrt = torch.diag(deg.pow(-0.5))
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        lap = deg_inv_sqrt @ adj @ deg_inv_sqrt  # normalized laplacian

        diff_x = lap @ lap @ x
        diff_x = diff_x[train_mask]

        self.weights = nn.Parameter(
            torch.linalg.inv(diff_x.T @ diff_x + 1 * torch.eye(diff_x.shape[1], device=device))
            @ diff_x.T
            @ torch.eye(self.out_channels, device=device)[y[train_mask]]
        )

    def forward(self, x, edge_index, mask=None):
        # TODO: make it completely sparse
        adj = pyg.utils.to_dense_adj(edge_index, max_num_nodes=x.shape[0])[0]
        if (adj.diagonal() == 0).all():
            adj = adj + torch.eye(x.shape[0], device=x.device)
        deg = adj.sum(1)
        deg_inv_sqrt = torch.diag(deg.pow(-0.5))
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        lap = deg_inv_sqrt @ adj @ deg_inv_sqrt

        diff_x = lap @ lap @ x
        if mask is not None:
            diff_x = diff_x[mask]

        out = diff_x @ self.weights
        return out

    def predict(self, dataset):
        """Predict on test set and return accuracy"""
        x = dataset.x
        edge_index = dataset.edge_index
        self.eval()
        _, pred = self(x,edge_index,).max(dim=1)
        correct = float(pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item())
        acc = correct / dataset.test_mask.sum().item()
        # print('Test Accuracy: {:.4f}'.format(acc))
        return acc


    def predict_train(self, dataset):
        """Predict on test set and return accuracy"""
        x = dataset.x
        edge_index = dataset.edge_index
        self.eval()
        _, pred = self(x,edge_index,).max(dim=1)
        correct = float(pred[dataset.train_mask].eq(dataset.y[dataset.train_mask]).sum().item())
        acc = correct / dataset.train_mask.sum().item()
        # print('Test Accuracy: {:.4f}'.format(acc))
        return acc

    def predict_valid(self, dataset):
        """Predict on test set and return accuracy"""
        x = dataset.x
        edge_index = dataset.edge_index
        self.eval()
        _, pred = self(x, edge_index, ).max(dim=1)
        correct = float(pred[dataset.val_mask].eq(dataset.y[dataset.val_mask]).sum().item())
        acc = correct / dataset.val_mask.sum().item()
        # print('Test Accuracy: {:.4f}'.format(acc))
        return acc



warnings.simplefilter(action='ignore', category=Warning)

parser = argparse.ArgumentParser(description="Network")
parser.add_argument('--dataset', default='Cora', help='Input dataset.')
dataset_name = parser.parse_args().dataset

# Print the parameters and pattern to verify
print(f"Dataset: {dataset_name}")

if dataset_name in dataset_params:
    params = dataset_params[dataset_name]
    num_epochs = params['num_epochs']
    lr = params['lr']
    weight_decay = params['weight_decay']
else:
    num_epochs = 200
    lr = 0.01
    weight_decay = 5e-4

# Load and preprocess the dataset
if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='dataset/', name=dataset_name, transform=T.NormalizeFeatures())
elif dataset_name in ['WikiCS', 'WikiCSX']:
    dataset = WikiCS(root='dataset/WikiCS', transform=T.NormalizeFeatures())
    config_path = f'./config/wikics.pkl'
elif dataset_name == 'Amazon':
    dataset = Amazon(root='dataset/Amazon', name='Computers', transform=T.NormalizeFeatures())
    raise NotImplementedError
elif dataset_name == 'Coauthor':
    dataset = Coauthor(root='dataset/Coauthor', name='CS', transform=T.NormalizeFeatures())
    raise NotImplementedError
else:
    raise ValueError(f"Dataset {dataset_name} is not supported.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data = dataset[0].to(device)

# Load train/valid/test split for non-Citation datasets
if dataset_name in ['Computers', 'Photo', 'Physics', 'WikiCS', 'WikiCSX', ]:
    with open(config_path, 'rb') as f:
        loaded_indices_dict = pickle.load(f)
        if dataset_name in ['WikiCS', 'WikiCSX', ]:
            assert calculate_md5_of_string(str(loaded_indices_dict)) == "ff62ecc913c95fba03412f445aae153f"
            split_id = loaded_indices_dict["split_id"]
            data.train_mask = data.train_mask[:, split_id].clone()
            data.val_mask = data.val_mask[:, split_id].clone()
            # Apply the normalization

            print("before standardize_features", data.x.min(), "to", data.x.max())
            data.x = standardize_features(data.x)
            print("after standardize_features", data.x.min(), "to", data.x.max())

            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

            train_size = train_mask.sum().item()
            val_size = val_mask.sum().item()
            test_size = test_mask.sum().item()

            # Print dataset sizes
            print(f"BEFORE: Train Mask:{train_mask.shape} Size: {train_size}")
            print(f"BEFORE: Validation Mask:{val_mask.shape} Size: {val_size}")
            print(f"BEFORE: Test Mask:{test_mask.shape} Size: {test_size}")

    data = set_masks_from_indices(data, loaded_indices_dict, device)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

train_size = train_mask.sum().item()
val_size = val_mask.sum().item()
test_size = test_mask.sum().item()

# Print dataset sizes
print(f"Train Mask:{train_mask.shape} Size: {train_size}")
print(f"Validation Mask:{val_mask.shape} Size: {val_size}")
print(f"Test Mask:{test_mask.shape} Size: {test_size}")

# Assertions to ensure no overlap between masks
assert (train_mask & val_mask).sum().item() == 0, "Train and Validation masks overlap!"
assert (val_mask & test_mask).sum().item() == 0, "Validation and Test masks overlap!"
assert (train_mask & test_mask).sum().item() == 0, "Train and Test masks overlap!"

# # Create inductive edge index (removing edges to val/test nodes)
# inductive_edge_index = []
# for src, tgt in data.edge_index.t().tolist():
#     if not (val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]):
#         inductive_edge_index.append([src, tgt])
# inductive_edge_index = torch.tensor(inductive_edge_index).t().contiguous()

indu_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
for i, (src, tgt) in enumerate(data.edge_index.t().tolist()):
    if val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]:
        indu_mask[i] = False
indu_mask = indu_mask.to(device)

# Prepare test and validation data
test_data = get_subgraph_data(data, data.test_mask)
val_data = get_subgraph_data(data, data.val_mask)

# Initial model training and evaluation
data_copy = data.clone()
data_copy = data_copy.to(device)
data_copy.edge_index = data_copy.edge_index[:, indu_mask]

model2 = SGC(dataset.num_features, dataset.num_classes)
model2.fit(edge_index=data_copy.edge_index, x=data_copy.x, y=data_copy.y, train_mask=data_copy.train_mask)
test_acc = model2.predict(test_data)
val_acc = model2.predict_valid(val_data)
train_acc = model2.predict_train(data_copy)
print("train_acc2", train_acc, "val_acc2", val_acc, "test_acc2", test_acc)


model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes, K=2).to(device)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data)
train_acc = model.predict_train(data_copy)
print("data shape", data_copy.x[train_mask].shape, data_copy.y[train_mask].shape)

# Print statistics
print("Max values per feature:", data_copy.x[train_mask].max())
print("Min values per feature:", data_copy.x[train_mask].min())

print("train_acc", train_acc, "val_acc", val_acc, "test_acc", test_acc)
model.fit(data_copy, num_epochs, lr, weight_decay)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data)
train_acc = model.predict_train(data_copy)
print("train_acc", train_acc, "val_acc", val_acc, "test_acc", test_acc)



# Distribution of classes in test and validation sets
output_dim = dataset.num_classes

# Calculate the total number of samples in each set
total_test_samples = data.test_mask.sum().item()
total_val_samples = data.val_mask.sum().item()

# Calculate the class distributions as counts
train_class_distribution = np.bincount(data.y[data.train_mask].cpu().numpy(), minlength=output_dim)
test_class_distribution = np.bincount(data.y[data.test_mask].cpu().numpy(), minlength=output_dim)
val_class_distribution = np.bincount(data.y[data.val_mask].cpu().numpy(), minlength=output_dim)

# Convert counts to percentages
test_class_percentages = (test_class_distribution / total_test_samples) * 100
val_class_percentages = (val_class_distribution / total_val_samples) * 100
train_class_percentages = (train_class_distribution / total_val_samples) * 100

# Print the distributions
print("Train Class Distribution (Percentages):")
for i, percentage in enumerate(train_class_percentages):
    print(f"Class {i}: {percentage:.2f}%")

# Print the distributions
print("Test Class Distribution (Percentages):")
for i, percentage in enumerate(test_class_percentages):
    print(f"Class {i}: {percentage:.2f}%")

print("\nValidation Class Distribution (Percentages):")
for i, percentage in enumerate(val_class_percentages):
    print(f"Class {i}: {percentage:.2f}%")
