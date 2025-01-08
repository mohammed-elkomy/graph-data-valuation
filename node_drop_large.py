"""
This script implements a dropping node experiment using the PC-Winter algorithm results.

It includes the following main components:
1. SGCNet: A SGC model used for downstream task evaluation
2. Data processing functions for graph data
3. PC-Winter value aggregation and node ranking
4. Node dropping experiment to evaluate the effectiveness of the valuation
"""

import collections
import os
import pickle
import re
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor
from torch_geometric.nn import SGConv

from pc_winter_run import calculate_md5_of_string, set_masks_from_indices

warnings.simplefilter(action='ignore', category=Warning)

# # Parameters
# dataset_name = 'Cora'  # Options: 'Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'Amazon', 'Coauthor'
# group_trunc_ratio_hop_1 = 0.5
# group_trunc_ratio_hop_2 = 0.7
# label_trunc_ratio = 0
# ratio = 20
# num_perms = 10

# Parameters
dataset_name = 'WikiCS'  # Options: 'Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'Amazon', 'Coauthor'
group_trunc_ratio_hop_1 = 0.7
group_trunc_ratio_hop_2 = 0.9
label_trunc_ratio = 0
ratio = 20
num_perms = 1

directory = 'value/'
pattern = re.compile(rf'^{dataset_name}_(\d+)_{num_perms}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value\.pkl$')


class SGCNet(nn.Module):
    """
    Simple Graph Convolutional Network model
    """

    def __init__(self, num_features, num_classes, seed=0):
        super(SGCNet, self).__init__()
        torch.manual_seed(seed)  # Set the seed for CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs
        self.conv = SGConv(num_features, num_classes, K=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_data):
        x, edge_index = input_data.x, input_data.edge_index
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, dataset, num_epochs=200):
        """Train the model"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(input_data)
            loss = F.nll_loss(out[input_data.train_mask], input_data.y[input_data.train_mask])
            loss.backward()
            optimizer.step()

    def predict(self, dataset):
        """Predict on test set and return accuracy"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float(pred[input_data.test_mask].eq(input_data.y[input_data.test_mask]).sum().item())
        acc = correct / input_data.test_mask.sum().item()
        print('Test Accuracy: {:.4f}'.format(acc))
        return acc

    def predict_valid(self, dataset):
        """Predict on validation set and return accuracy"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float(pred[input_data.val_mask].eq(input_data.y[input_data.val_mask]).sum().item())
        acc = correct / input_data.val_mask.sum().item()
        print('Validation Accuracy: {:.6f}'.format(acc))
        return acc


def get_subgraph_data(data, mask):
    """Extract subgraph data based on the given mask"""
    nodes = mask.nonzero().view(-1)
    edge_mask_src = (data.edge_index[0].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask_dst = (data.edge_index[1].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask = edge_mask_src & edge_mask_dst
    sub_edge_index = data.edge_index[:, edge_mask]

    test_mask = data.test_mask
    val_mask = data.val_mask

    sub_data = Data(x=data.x, edge_index=sub_edge_index, y=data.y, test_mask=test_mask, val_mask=val_mask)
    return sub_data


# Find matching files for PC-Winter results
matching_files = []
for filename in os.listdir(directory):
    if pattern.match(filename):
        matching_files.append(filename)
filenames = matching_files[:ratio]
ratio = min(len(filenames), ratio)  # Limit the number of files to process
print(f"Processing the files: {filenames}\n")

# Extract and aggregate PC-Winter values
results = collections.defaultdict(list)
counter = 0
for filename in filenames:
    with open(os.path.join(directory, filename), 'rb') as f:
        data = pickle.load(f)
    for key, sub_dict in data.items():
        for sub_key, sub_sub_dict in sub_dict.items():
            for sub_sub_key, value in sub_sub_dict.items():
                results[(key, sub_key, sub_sub_key)].append(value)
    counter += 1

# Average the values
for key, values in results.items():
    results[key] = sum(values) / (len(values) * num_perms)

# Convert to DataFrame
data = [{'key1': k1, 'key2': k2, 'key3': k3, 'value': v} for (k1, k2, k3), v in results.items()]
win_df = pd.DataFrame(data)

# Aggregate values for different hop levels
win_df_11 = pd.DataFrame(win_df[win_df['key2'].isin(win_df['key1']) == False].groupby('key2').value.sum().sort_values()).reset_index()
win_df_11.columns = ['key', 'value']
hop_1_list = win_df[win_df['key2'].isin(win_df['key1']) == False]['key2'].unique()
win_df_12 = pd.DataFrame(win_df[(win_df['key3'] != win_df['key2']) & (win_df['key3'].isin(hop_1_list))].groupby('key3').value.sum().sort_values()).reset_index()
win_df_12.columns = ['key', 'value']

win_df_1 = pd.DataFrame(pd.concat([win_df_11, win_df_12]).groupby('key').value.sum().sort_values()).reset_index()
win_df_2 = pd.DataFrame(win_df[(win_df['key3'].isin(win_df['key2']) == False) & (win_df['key3'].isin(win_df['key1']) == False)].groupby('key3').value.sum().sort_values()).reset_index()
win_df_2.columns = ['key', 'value']

# Combine and sort unlabeled nodes
unlabled_win_df = pd.concat([win_df_1, win_df_2])
unlabled_win_df = unlabled_win_df.sort_values('value', ascending=False)
unlabeled_win = torch.tensor(unlabled_win_df['key'].values)
unlabeled_win_value = unlabled_win_df['value'].values


# Load and preprocess the dataset
if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='dataset/', name=dataset_name, transform=T.NormalizeFeatures())
elif dataset_name == 'WikiCS':
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
if dataset_name in ['Computers', 'Photo', 'Physics', 'WikiCS']:
    with open(config_path, 'rb') as f:
        loaded_indices_dict = pickle.load(f)
        if dataset_name == 'WikiCS':
            assert calculate_md5_of_string(str(loaded_indices_dict)) == "ff62ecc913c95fba03412f445aae153f"
            split_id = loaded_indices_dict["split_id"]
            data.train_mask = data.train_mask[:, split_id].clone()
            data.val_mask = data.val_mask[:, split_id].clone()

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

# Node dropping experiment
win_acc = []
val_acc_list = []
node_list = unlabeled_win.cpu().numpy()
drop_num = len(node_list) + 1

# Initial model training and evaluation
data_copy = data.clone()
data_copy = data_copy.to(device)
data_copy.edge_index = data_copy.edge_index[:, indu_mask]

model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
model.fit(data_copy)
test_acc = model.predict(test_data)
val_acc = model.predict_valid(val_data)
win_acc += [test_acc]
val_acc_list += [val_acc]

# Iteratively drop nodes and evaluate
for j in range(1, drop_num):
    cur_player = node_list[j - 1]
    print('cur_player: ', cur_player)
    cur_node_list = node_list[:j]

    edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
    for node in cur_node_list:
        edge_mask[data.edge_index[0] == node] = False
        edge_mask[data.edge_index[1] == node] = False

    edge_mask = edge_mask & indu_mask
    data_copy = data.clone()
    data_copy = data_copy.to(device)
    data_copy.edge_index = data_copy.edge_index[:, edge_mask]

    model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
    model.fit(data_copy)
    test_acc = model.predict(test_data)
    val_acc = model.predict_valid(val_data)
    win_acc += [test_acc]
    val_acc_list += [val_acc]

# Save results
path = 'res/'
os.makedirs(path, exist_ok=True)
with open(os.path.join(path, f'node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{counter}_{dataset_name}_test.pkl'), 'wb') as file:
    pickle.dump(win_acc, file)

with open(os.path.join(path, f'node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{counter}_{dataset_name}_vali.pkl'), 'wb') as file:
    pickle.dump(val_acc_list, file)
