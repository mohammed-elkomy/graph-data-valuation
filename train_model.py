"""
This script trains a model on WikiCS for a number of epochs with the full data
This shows a single model retraining from the node dropping experiment
"""
import argparse
import pickle
import warnings

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, WikiCS

from node_drop_large import SGCNet, get_subgraph_data
from pc_winter_run import calculate_md5_of_string, set_masks_from_indices, dataset_params

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
elif dataset_name in ['WikiCS', 'WikiCS2']:
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
if dataset_name in ['Computers', 'Photo', 'Physics', 'WikiCS', 'WikiCS2']:
    with open(config_path, 'rb') as f:
        loaded_indices_dict = pickle.load(f)
        if dataset_name in ['WikiCS', 'WikiCS2']:
            assert calculate_md5_of_string(str(loaded_indices_dict)) == "ff62ecc913c95fba03412f445aae153f"
            split_id = loaded_indices_dict["split_id"]
            data.train_mask = data.train_mask[:, 3].clone()
            data.val_mask = data.val_mask[:, 3].clone()

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

    # data = set_masks_from_indices(data, loaded_indices_dict, device)

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

# Create inductive edge index (removing edges to val/test nodes)
inductive_edge_index = []
for src, tgt in data.edge_index.t().tolist():
    if not (val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]):
        inductive_edge_index.append([src, tgt])
inductive_edge_index = torch.tensor(inductive_edge_index).t().contiguous()

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

model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes,K=6).to(device)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data)
print(test_acc, val_acc)
model.fit(data_copy, num_epochs*5, lr, weight_decay)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data)
print(test_acc, val_acc)
