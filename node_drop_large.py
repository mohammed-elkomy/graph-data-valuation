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
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor
from torch_geometric.nn import SGConv
from tqdm import tqdm

import argparse
from pc_winter_run import calculate_md5_of_string, set_masks_from_indices, dataset_params

warnings.simplefilter(action='ignore', category=Warning)

WORKERS = 10
directory = 'value/'


def parse_args():
    """
    Parses command-line arguments for the script and returns the parsed parameters.

    Returns:
        args: Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script to run graph dataset experiments with specified parameters.")

    # Define command-line arguments
    parser.add_argument('--dataset_name', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'WikiCS2','Amazon', 'Coauthor'],
                        required=True, help="Dataset name. Options: 'Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'Amazon', 'Coauthor'.")
    parser.add_argument('--group_trunc_ratio_hop_1', type=float, required=True,
                        help="Group truncation ratio for hop 1.")
    parser.add_argument('--group_trunc_ratio_hop_2', type=float, required=True,
                        help="Group truncation ratio for hop 2.")
    parser.add_argument('--label_trunc_ratio', type=float, required=True,
                        help="Label truncation ratio.")
    parser.add_argument('--ratio', type=int, required=True,
                        help="Ratio parameter.")
    parser.add_argument('--num_perms', type=int, required=True,
                        help="Number of permutations.")
    parser.add_argument('--parallel_idx', type=int, required=True,
                        help="Index for parallel execution.")
    parser.add_argument('--min_occ_perc', type=int, default=70,
                        help="Minimum occurrences (based on percentiles) for each node during the pc-winter value approximation")

    return parser.parse_args()


def format_ratio(value):
    """Converts a float value to both string representations of `0` and `0.0`."""
    return r'(0|0\.0)' if value == 0 else str(value).replace('.', r'\.')


class SGCNet(nn.Module):
    """
    Simple Graph Convolutional Network model
    """

    def __init__(self, num_features, num_classes, K=2, seed=0):
        super(SGCNet, self).__init__()
        torch.manual_seed(seed)  # Set the seed for CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs
        self.conv = SGConv(num_features, num_classes, K=K)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_data):
        x, edge_index = input_data.x, input_data.edge_index
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, dataset, num_epochs, lr, weight_decay):
        """Train the model"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        # print('Test Accuracy: {:.4f}'.format(acc))
        return acc

    def predict_valid(self, dataset):
        """Predict on validation set and return accuracy"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float(pred[input_data.val_mask].eq(input_data.y[input_data.val_mask]).sum().item())
        acc = correct / input_data.val_mask.sum().item()
        # print('Validation Accuracy: {:.6f}'.format(acc))
        return acc

    def predict_train(self, dataset):
        """Predict on test set and return accuracy"""
        model = self.to(self.device)
        input_data = dataset.to(self.device)
        model.eval()
        _, pred = model(input_data).max(dim=1)
        correct = float(pred[input_data.train_mask].eq(input_data.y[input_data.train_mask]).sum().item())
        acc = correct / input_data.train_mask.sum().item()
        # print('Test Accuracy: {:.4f}'.format(acc))
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


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Set the parameters from parsed arguments
    dataset_name = args.dataset_name
    group_trunc_ratio_hop_1 = args.group_trunc_ratio_hop_1
    group_trunc_ratio_hop_2 = args.group_trunc_ratio_hop_2
    label_trunc_ratio = args.label_trunc_ratio
    ratio = args.ratio
    num_perms = args.num_perms
    parallel_idx = args.parallel_idx
    min_occ_perc = args.min_occ_perc

    assert parallel_idx < WORKERS

    # Prepare the regex pattern
    label_trunc_ratio_str = format_ratio(label_trunc_ratio)
    group_trunc_ratio_hop_1_str = format_ratio(group_trunc_ratio_hop_1)
    group_trunc_ratio_hop_2_str = format_ratio(group_trunc_ratio_hop_2)

    # Construct the pattern
    value_pattern = re.compile(
        rf'^{dataset_name}_(\d+)_'
        rf'{num_perms}_'
        rf'{label_trunc_ratio_str}_'
        rf'{group_trunc_ratio_hop_1_str}_'
        rf'{group_trunc_ratio_hop_2_str}_pc_value\.pkl$'
    )

    # Construct the pattern
    count_pattern = re.compile(
        rf'^{dataset_name}_(\d+)_'
        rf'{num_perms}_'
        rf'{label_trunc_ratio_str}_'
        rf'{group_trunc_ratio_hop_1_str}_'
        rf'{group_trunc_ratio_hop_2_str}_pc_value_count\.pkl$'
    )

    # Print the parameters and pattern to verify
    print(f"Dataset: {dataset_name}")
    print(f"Group Truncation Ratio Hop 1: {group_trunc_ratio_hop_1}")
    print(f"Group Truncation Ratio Hop 2: {group_trunc_ratio_hop_2}")
    print(f"Label Truncation Ratio: {label_trunc_ratio}")
    print(f"Ratio: {ratio}")
    print(f"Number of Permutations: {num_perms}")
    print(f"Parallel Index: {parallel_idx}")
    print(f"Regex Pattern: {value_pattern.pattern}")

    if dataset_name in dataset_params:
        params = dataset_params[dataset_name]
        num_epochs = params['num_epochs']
        lr = params['lr']
        weight_decay = params['weight_decay']
    else:
        num_epochs = 200
        lr = 0.01
        weight_decay = 5e-4

    # Find matching files for PC-Winter results
    value_matching_files = []
    count_matching_files = []
    for filename in os.listdir(directory):
        if value_pattern.match(filename):
            value_matching_files.append(filename)
        if count_pattern.match(filename):
            count_matching_files.append(filename)

    value_filenames = value_matching_files[:ratio]
    count_filenames = count_matching_files[:ratio]
    ratio = min(len(value_filenames), ratio)  # Limit the number of files to process
    print(f"Processing the files {len(value_filenames)}: {value_filenames}\n")
    print(f"Processing the files {len(count_filenames)}: {count_filenames}\n")

    # Extract and aggregate PC-Winter values
    results = collections.defaultdict(list)
    counts = collections.defaultdict(int)

    for filename in value_filenames:
        with open(os.path.join(directory, filename), 'rb') as f:
            data = pickle.load(f)
        for key, sub_dict in data.items():
            for sub_key, sub_sub_dict in sub_dict.items():
                for sub_sub_key, value in sub_sub_dict.items():
                    results[(key, sub_key, sub_sub_key)].append(value)

    for filename in count_filenames:
        with open(os.path.join(directory, filename), 'rb') as f:
            data = pickle.load(f)
        for key, sub_dict in data.items():
            for sub_key, sub_sub_dict in sub_dict.items():
                for sub_sub_key, value in sub_sub_dict.items():
                    counts[key] += value
                    counts[sub_key] += value
                    counts[sub_sub_key] += value

    # Average the values
    for key, values in results.items():
        results[key] = sum(values) / (len(values) * num_perms)  # TODO is it right to divide by num_perms?

    # Convert to DataFrame
    data = [{'key1': k1, 'key2': k2, 'key3': k3, 'value': v} for (k1, k2, k3), v in results.items()]
    win_df = pd.DataFrame(data)
    # labeled > hop 1 > hop 2
    # Aggregate values for different hop levels
    win_df_11 = pd.DataFrame(win_df[win_df['key2'].isin(win_df['key1']) == False].groupby('key2').value.sum().sort_values()).reset_index()
    win_df_11.columns = ['key', 'value']  # unlabelled nodes from hop 1

    hop_1_list = win_df[win_df['key2'].isin(win_df['key1']) == False]['key2'].unique()
    win_df_12 = pd.DataFrame(win_df[(win_df['key3'] != win_df['key2']) & (win_df['key3'].isin(hop_1_list))].groupby('key3').value.sum().sort_values()).reset_index()
    win_df_12.columns = ['key', 'value']  # unlabelled nodes from hop 1 with contributions coming from hop 2

    win_df_1 = pd.DataFrame(pd.concat([win_df_11, win_df_12]).groupby('key').value.sum().sort_values()).reset_index()  # full winter value for unlabelled nodes from hop 1

    win_df_2 = pd.DataFrame(win_df[(win_df['key3'].isin(win_df['key2']) == False) & (win_df['key3'].isin(win_df['key1']) == False)].groupby('key3').value.sum().sort_values()).reset_index()
    win_df_2.columns = ['key', 'value']  # unlabelled nodes from hop 2 (leaf nodes; no further contribution)

    # Combine and sort unlabeled nodes
    unlabled_win_df = pd.concat([win_df_1, win_df_2])
    unlabled_win_df = unlabled_win_df.sort_values('value', ascending=False)
    # count cut off
    print("before count filtration", unlabled_win_df.shape)
    unlabled_win_df['count'] = unlabled_win_df['key'].map(counts)
    print(unlabled_win_df['count'].value_counts())
    percentile_threshold = np.percentile(unlabled_win_df['count'], min_occ_perc)
    # Filter rows where 'count' is greater than the xth percentile (min_occ_perc)
    unlabled_win_df = unlabled_win_df[unlabled_win_df["count"] > percentile_threshold]
    print("after count filtration", unlabled_win_df.shape)

    unlabeled_win = torch.tensor(unlabled_win_df['key'].values)
    unlabeled_win_value = unlabled_win_df['value'].values

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

    drop_num = len(node_list)
    parallel_subset = len(node_list) // WORKERS

    if parallel_idx == -1:
        start_sim = 1
        end_sim = len(node_list)
        parallel_idx = 0  # for the file naming
    else:
        start_sim = parallel_subset * parallel_idx
        end_sim = (parallel_subset) * (parallel_idx + 1)

    if start_sim == 0:
        # Initial model training and evaluation
        data_copy = data.clone()
        data_copy = data_copy.to(device)
        data_copy.edge_index = data_copy.edge_index[:, indu_mask]

        model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
        model.fit(data_copy, num_epochs, lr, weight_decay)
        test_acc = model.predict(test_data)
        val_acc = model.predict_valid(val_data)
        win_acc += [test_acc]
        val_acc_list += [val_acc]
        start_sim = 1

    print("starting at id", start_sim, "to", end_sim)
    # Iteratively drop nodes and evaluate
    for j in tqdm(range(start_sim, end_sim)):
        # nodes are sorted according to their scores in descending order
        cur_player = node_list[j - 1]
        print('cur_player: ', cur_player)
        cur_node_list = node_list[:j]
        print(data.edge_index.shape[1])
        exit()
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
        for node in cur_node_list:
            edge_mask[data.edge_index[0] == node] = False
            edge_mask[data.edge_index[1] == node] = False

        edge_mask = edge_mask & indu_mask
        data_copy = data.clone()
        data_copy = data_copy.to(device)
        data_copy.edge_index = data_copy.edge_index[:, edge_mask]

        model = SGCNet(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
        model.fit(data_copy, num_epochs, lr, weight_decay)
        test_acc = model.predict(test_data)
        val_acc = model.predict_valid(val_data)
        win_acc += [test_acc]
        val_acc_list += [val_acc]

    # Save results
    path = 'res/'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'{parallel_idx}-node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{ratio}_{dataset_name}_test.pkl'), 'wb') as file:
        pickle.dump(win_acc, file)

    with open(os.path.join(path, f'{parallel_idx}-node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_{ratio}_{dataset_name}_vali.pkl'), 'wb') as file:
        pickle.dump(val_acc_list, file)
