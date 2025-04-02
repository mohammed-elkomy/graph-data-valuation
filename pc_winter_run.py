"""
This script implements the PC-Winter (Precedence-Constrained Winter) algorithm for graph data valuation.
Key features of the implementation include:

1. Evaluation Model: SGC (Simple Graph Convolution) is used, split into two parts:
   a) Local propagation: Implemented in generate_features_and_labels_ind function.
   b) Classifier training: Using an MLP (Multi-Layer Perceptron).

2. Local Propagation: generate_features_and_labels_ind function implements the local propagation strategy,
   combining previously propagated node features with partially propagated features of the target node.

3. Preorder Traversal: The main nested loops implement the preorder traversal of the contribution tree,
   a key component of the PC-Winter algorithm.

4. Hierarchical Truncation: Implemented to reduce computational complexity,
   truncating at both the 1-hop node level and the 2-hop node levels.

5. Inductive Setting: The script sets up an inductive learning environment by removing validation and test nodes
   from the training graph.

6. Value Accumulation: Throughout the traversal, values are accumulated for each node,
   representing their contribution to the model's performance.

The script uses command-line arguments to control algorithm behavior, including dataset selection,
number of hops, random seed, number of permutations, and truncation ratios.
"""
import base64
import hashlib
import itertools
import json
import os
import pathlib
import sys
from logging import Logger

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import accuracy_score
import random
from torch_geometric.transforms import RootedEgoNets
from torch_geometric.utils import k_hop_subgraph
import torch_geometric
import time
from itertools import chain, combinations
import numpy as np
import pickle
import pandas as pd
import argparse
import torch.nn.functional as F
import collections
import copy
import torch_geometric.transforms as T
from torch.nn.functional import cosine_similarity
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, WikiCS

from typing import Optional
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from tqdm import tqdm
import subprocess

dataset_params = {
    'Computers': {
        'num_epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4
    },
    'Photo': {
        'num_epochs': 200,
        'lr': 0.01,
        'weight_decay': 5e-4
    },
    'Physics': {
        'num_epochs': 30,
        'lr': 0.01,
        'weight_decay': 5e-4
    },
    'WikiCS': {
        'num_epochs': 500,
        'lr': 0.05,
        'weight_decay': 5e-4
    },
    'WikiCSX': {
        'num_epochs': 30,
        'lr': 0.01,
        'weight_decay': 5e-4
    },
}


class SGConvNoWeight(MessagePassing):
    """
    The modified SGConv operator without the trainable linear layer.
    This class implements the feature propagation mechanism used in the local propagation strategy.
    """

    def __init__(self, K: int = 2,
                 cached: bool = False, add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, K={self.K})')


def standardize_features(x):
    # Calculate the mean of each row
    row_mean = x.mean(dim=1, keepdim=True)
    # Calculate the standard deviation of each row
    row_std = x.std(dim=1, keepdim=True)
    # Prevent division by zero by setting any zero std values to 1
    row_std[row_std == 0] = 1
    # Standardize each row: (x - mean) / std
    return (x - row_mean) / row_std


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        return F.log_softmax(out, dim=1)

    def predict(self, x):
        output = self.forward(x)
        return output

    def fit(self, X, y, val_X, val_y, num_iter=200, lr=0.01, weight_decay=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        counter = 0

        for epoch in range(num_iter):
            self.train()
            optimizer.zero_grad()
            output = self(X)
            loss = F.nll_loss(output, y)
            # if epoch % 50 == 0:
            #     print("loss", loss.item())
            loss.backward()
            optimizer.step()


def adjacency_to_edge_list(adj_matrix):
    """
    Convert an adjacency matrix to an edge list representation.
    """
    # Convert the adjacency matrix to COO format
    source, target = adj_matrix.nonzero().unbind(dim=1)
    # Create the edge_index tensor
    edge_index = torch.stack([source, target], dim=0)
    return edge_index


def stack_torch_tensors(input_tensors):
    unrolled = [input_tensors[k].view(-1, 1) for k in range(len(input_tensors))]
    return torch.cat(unrolled)


def generate_features_and_labels_ind(cur_hop_1_list, cur_hop_2_list, cur_labeled_list, target_node, node_map, ind_edge_index, data, device):
    """
    This function implements the local propagation strategy for the PC-Winter algorithm.
    It generates features and labels for the inductive setting, considering the current state
    of the graph during a permutation.

    The key step is the concatenation of previously propagated features with the
    partially propagated features of the current target node:
    train_features = torch.cat((X_ind_propogated[cur_labeled_list], X_ind_propogated_temp_[target_node].unsqueeze(0)), dim=0)
    train_labels = torch.cat((data.y[cur_labeled_list], data.y[target_node].unsqueeze(0)), dim=0)

    This approach allows for efficient computation of node contributions in the PC-Winter algorithm.
    """
    # cur_hop_1_list = list(node_map[target_node])
    # cur_hop_2_list = list(node_map[target_node][list(node_map[target_node])[-1]].keys())
    start = time.time()
    A_temp = torch.zeros((data.x.size(0), data.x.size(0)), device=device)
    A_temp[ind_edge_index[0], ind_edge_index[1]] = 1

    mask = torch.zeros_like(A_temp)
    cur_hop_1_list_torch = torch.tensor(cur_hop_1_list)
    mask[target_node, cur_hop_1_list] = 1
    mask[cur_hop_1_list, target_node] = 1
    if len(cur_hop_1_list) > 1:
        for hop_1_node in cur_hop_1_list[:-1]:
            hop_2_list = list(node_map[target_node][hop_1_node].keys())
            mask[hop_1_node, hop_2_list] = 1
            mask[hop_2_list, hop_1_node] = 1
        mask[cur_hop_1_list[-1], cur_hop_2_list] = 1
        mask[cur_hop_2_list, cur_hop_1_list[-1]] = 1
    else:
        mask[cur_hop_1_list[-1], cur_hop_2_list] = 1
        mask[cur_hop_2_list, cur_hop_1_list[-1]] = 1

    conv = SGConvNoWeight(K=2)
    cur_edge_index = adjacency_to_edge_list(mask)
    X_ind_propogated_temp_ = conv(data.x, cur_edge_index)

    train_features = torch.cat((X_ind_propogated[cur_labeled_list], X_ind_propogated_temp_[target_node].unsqueeze(0)), dim=0)
    train_labels = torch.cat((data.y[cur_labeled_list], data.y[target_node].unsqueeze(0)), dim=0)

    return train_features, train_labels


def evaluate_retrain_model(model_class, num_features, num_classes, train_features, train_labels, val_features, val_labels, device, num_iter=200, lr=0.01, weight_decay=5e-4):
    """
    This function creates, trains, and evaluates a model on the given data.
    It's used to compute the utility function in the PC-Winter algorithm.
    The utility is measured as the validation accuracy of the trained model.
    """
    # Create and train the model
    model = model_class(num_features, num_classes).to(device)
    model.fit(train_features, train_labels, val_features, val_labels, num_iter=num_iter, lr=lr, weight_decay=weight_decay)
    # Make predictions on the validation set
    predictions = model(val_features)
    # Calculate the accuracy of the model
    val_acc = (predictions.argmax(dim=1) == val_labels).float().mean().item()

    return val_acc


def generate_maps(train_idx_list, num_hops, edge_index):
    """
    This function generates the necessary data structures for efficient computation
    of the PC-Winter algorithm, including the labeled_to_player_map which represents
    the contribution tree structure.

    The key chain stands for one contribution path of a node in a computational tree:
    [labeled][labeled][labeled] is a labeled node;
    [labeled][hop_1_node][hop_1_node] is a label' node's 1-distance neighbor;
    [labeled][hop_1_node][hop_2_node] is a label' node's 2-distance neighbor;
    Here the key index is the node index in the graph.

    [labeled][labeled][hop_1_node] -> this is not possible in this mapping
    """

    labeled_to_player_map = {}
    sample_value_dict = {}
    sample_counter_dict = {}

    for labeled in train_idx_list:
        hop_1_nodes, _, _, _ = k_hop_subgraph(int(labeled), num_hops=1, edge_index=edge_index, relabel_nodes=False)
        hop_1_nodes_list = list(hop_1_nodes.cpu().numpy())
        hop_1_nodes_list.remove(labeled)
        labeled_to_player_map[labeled] = {}
        sample_value_dict[labeled] = {}
        sample_counter_dict[labeled] = {}
        labeled_to_player_map[labeled][labeled] = {}
        sample_value_dict[labeled][labeled] = {}
        sample_counter_dict[labeled][labeled] = {}

        for hop_1_node in hop_1_nodes_list:
            sub_nodes_2, _, _, _ = k_hop_subgraph(int(hop_1_node), num_hops=1, edge_index=edge_index, relabel_nodes=False)
            sub_nodes_2_list = list(sub_nodes_2.cpu().numpy())
            sub_nodes_2_list.remove(hop_1_node)
            labeled_to_player_map[labeled][hop_1_node] = {}
            sample_value_dict[labeled][hop_1_node] = {}
            sample_counter_dict[labeled][hop_1_node] = {}

            for hop_2_node in sub_nodes_2_list:
                labeled_to_player_map[labeled][hop_1_node][hop_2_node] = [hop_2_node]  # correspond to hop-2 marginal cont
                sample_value_dict[labeled][hop_1_node][hop_2_node] = 0
                sample_counter_dict[labeled][hop_1_node][hop_2_node] = 0

            labeled_to_player_map[labeled][hop_1_node][hop_1_node] = [hop_1_node]  # Self-loop without full subgraph (correspond to hop-1 marginal cont)
            sample_value_dict[labeled][hop_1_node][hop_1_node] = 0
            sample_counter_dict[labeled][hop_1_node][hop_1_node] = 0

        labeled_to_player_map[labeled][labeled][labeled] = [labeled]  # Self-loop without full subgraph (correspond to hop-0 marginal cont)
        sample_value_dict[labeled][labeled][labeled] = 0
        sample_counter_dict[labeled][labeled][labeled] = 0
        # this way of modelling allows LWW to happen but LLW is not allowed # L is labeled hop-0, W is hop-1

    return labeled_to_player_map, sample_value_dict, sample_counter_dict


def get_subgraph_data(data_edge_index, mask):
    """
    This function extracts a subgraph from the given graph based on a mask.
    The resulting subgraph only contains edges between nodes in the mask.
    """

    # Nodes to be considered
    edge_index = data_edge_index.clone()
    nodes = mask.nonzero().view(-1)

    # Extract the edges for these nodes
    edge_mask_src = (edge_index[0].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask_dst = (edge_index[1].unsqueeze(-1) == nodes.unsqueeze(0)).any(dim=-1)
    edge_mask = edge_mask_src & edge_mask_dst

    sub_edge_index = edge_index[:, edge_mask]
    return sub_edge_index


def propagate_features(edge_index, node_features):
    """SGC propagation of node features using the given edge_index."""
    A = torch.zeros((node_features.size(0), node_features.size(0)), device=device)  # adj mat
    A[edge_index[0], edge_index[1]] = 1
    A_hat = A + torch.eye(A.size(0), device=device)  # self loops
    D_hat_diag = A_hat.sum(dim=1).pow(-0.5)
    D_hat = torch.diag(D_hat_diag)  # degree inverse in diag
    L_norm = D_hat.mm(A_hat).mm(D_hat)  # normalized graph Laplacian
    return L_norm.mm(L_norm.mm(node_features))  # no non-linearities ; number of layers is limited


def set_masks_from_indices(data, indices_dict, device):
    """
    Set train, validation, and test masks for the graph data based on provided indices.
    """

    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=bool).to(device)
    train_mask[indices_dict["train"]] = 1
    val_mask = torch.zeros(num_nodes, dtype=bool).to(device)
    val_mask[indices_dict["val"]] = 1
    test_mask = torch.zeros(num_nodes, dtype=bool).to(device)
    test_mask[indices_dict["test"]] = 1

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument('--dataset', default='Cora', help='Input dataset.')
    parser.add_argument('--num_hops', type=int, default=2, help='Number of hops.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for permutation.')
    parser.add_argument('--num_perm', type=int, default=10, help='Number of permutations.')
    parser.add_argument('--wg_l1', type=int, required=True,
                        help="count for level 1 within groups")
    parser.add_argument('--wg_l2', type=int, required=True,
                        help="count for level 2 within groups")
    parser.add_argument('--label_trunc_ratio', type=float, default=0, help='Label trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_1', type=float, default=0.5, help='Hop 1 Group trunc ratio')
    parser.add_argument('--group_trunc_ratio_hop_2', type=float, default=0.7, help='Hop 2 Group trunc ratio.')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--exp_id', type=str, required=True)

    return parser.parse_args()


def calculate_md5_of_string(input_string):
    """Calculate the MD5 checksum of a given string."""
    md5_hash = hashlib.md5()  # Create an MD5 hash object
    md5_hash.update(input_string.encode('utf-8'))  # Encode the string and update the hash
    return md5_hash.hexdigest()  # Return the hexadecimal digest


def generate_wikics_split(data, seed=42):
    """
    function for creating the wiki-cs split (fixed across experiments)
    """
    num_per_class = 20
    split_id = 0
    val_test_perc = 0.25

    # num_per_class = 1
    # split_id = 0
    # val_test_perc = 0.001

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get one of the split masks
    train_mask = data.train_mask[:, split_id].clone()
    val_mask = data.val_mask[:, split_id].clone()
    test_mask = data.test_mask.clone()  # Full test mask with no split

    # Initialize new masks
    num_classes = data.y.max().item() + 1
    new_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    new_val_mask = torch.zeros_like(val_mask, dtype=torch.bool)
    new_test_mask = torch.zeros_like(test_mask, dtype=torch.bool)

    # Select 20 samples per class for training
    for c in range(num_classes):
        class_indices = (data.y == c).nonzero(as_tuple=True)[0]
        class_train_indices = class_indices[train_mask[class_indices]]
        selected_train_indices = class_train_indices[torch.randperm(class_train_indices.size(0))[:num_per_class]]
        new_train_mask[selected_train_indices] = True

    # Get only 15% of data from the original val_mask and test_mask
    val_indices = val_mask.nonzero(as_tuple=True)[0]
    test_indices = test_mask.nonzero(as_tuple=True)[0]

    num_val = int(val_test_perc * val_indices.size(0))
    num_test = int(val_test_perc * test_indices.size(0))

    # Shuffle and select the required number of samples
    selected_val_indices = val_indices[torch.randperm(val_indices.size(0))[:num_val]]
    selected_test_indices = test_indices[torch.randperm(test_indices.size(0))[:num_test]]

    new_val_mask[selected_val_indices] = True
    new_test_mask[selected_test_indices] = True

    # Class distribution counters
    print("\nClass distribution per split:")
    for c in range(num_classes):
        train_count = (new_train_mask & (data.y == c)).sum().item()
        val_count = (new_val_mask & (data.y == c)).sum().item()
        test_count = (new_test_mask & (data.y == c)).sum().item()
        print(f"Class {c}: Train: {train_count}, Val: {val_count}, Test: {test_count}")

    split_config = {
        "train": list((new_train_mask.nonzero(as_tuple=True)[0]).cpu().numpy()),
        "val": list((new_val_mask.nonzero(as_tuple=True)[0]).cpu().numpy()),
        "test": list((new_test_mask.nonzero(as_tuple=True)[0]).cpu().numpy()),
        "split_id": split_id,
    }

    print(new_train_mask.sum(), new_val_mask.sum(), new_test_mask.sum())
    with open(f"config/wikics.pkl", "wb") as f:
        pickle.dump(split_config, f)

    print("hash for configs", calculate_md5_of_string(str(split_config)))
    return data


def pc_winter(wg_l1, wg_l2, max_model_retrainings=10000, verbose=False):
    eval_count = 0
    eval_progress = tqdm(total=max_model_retrainings, desc="Evaluations Progress")
    permutations = [
        [91, 5, 83, 65, 139, 19, 12, 2, 26, 60, 105, 74, 111, 117, 25, 108, 16, 115, 98, 24, 101, 41, 34, 78, 1, 18, 96, 35, 29, 20, 47, 61, 68, 7, 10, 120, 13, 15, 125, 100, 27, 131, 114, 84, 43, 77, 99, 123, 6, 39, 63, 97, 93, 64, 80, 9, 14, 92, 133, 31, 72, 69, 11, 107, 4, 110, 54, 86, 37, 132, 51, 88, 127, 53, 119, 90, 71, 70, 113, 85, 75, 82, 62, 67, 22, 23, 76, 48, 59, 116, 89, 81, 17, 126, 138, 95, 79, 121, 28, 128, 33, 45, 42, 40, 32, 136, 134, 112, 49, 8, 30, 46, 66, 56, 102, 124, 73, 118, 21, 130, 0, 3, 52, 106, 38, 44, 135, 36, 137, 57, 55, 94, 109, 103, 58, 50, 87, 104, 129, 122],
        [75, 63, 109, 13, 0, 131, 6, 128, 33, 98, 121, 56, 68, 45, 18, 10, 111, 92, 24, 124, 58, 14, 9, 86, 67, 133, 138, 101, 78, 43, 120, 100, 46, 1, 80, 113, 114, 17, 11, 47, 39, 118, 87, 69, 61, 41, 84, 42, 88, 122, 52, 119, 136, 31, 139, 99, 95, 60, 105, 16, 85, 59, 72, 64, 82, 30, 71, 103, 107, 135, 116, 91, 117, 129, 93, 94, 48, 83, 55, 19, 65, 66, 54, 20, 112, 134, 44, 123, 8, 73, 97, 137, 21, 102, 23, 132, 53, 125, 76, 130, 34, 106, 126, 15, 50, 108, 35, 57, 81, 70, 110, 26, 96, 12, 3, 104, 4, 49, 77, 89, 115, 27, 7, 62, 51, 90, 79, 127, 38, 40, 28, 22, 2, 36, 29, 74, 5, 37, 25, 32],
        [68, 46, 7, 32, 72, 114, 6, 52, 4, 99, 116, 17, 59, 136, 26, 122, 81, 137, 86, 87, 117, 30, 94, 106, 88, 5, 90, 109, 11, 3, 66, 47, 51, 15, 35, 130, 57, 20, 101, 84, 132, 14, 64, 91, 55, 110, 78, 29, 85, 74, 53, 100, 103, 24, 89, 75, 61, 33, 105, 0, 95, 13, 126, 111, 50, 135, 25, 39, 112, 49, 92, 62, 2, 134, 71, 113, 8, 34, 36, 80, 115, 70, 67, 45, 102, 22, 107, 108, 44, 139, 129, 83, 77, 76, 37, 119, 42, 82, 133, 73, 120, 79, 56, 18, 48, 27, 54, 43, 96, 1, 121, 124, 104, 58, 63, 123, 9, 97, 60, 65, 12, 41, 31, 10, 28, 127, 19, 21, 138, 128, 125, 38, 118, 16, 131, 98, 93, 40, 69, 23],
        [104, 67, 27, 101, 22, 126, 26, 105, 71, 31, 54, 11, 86, 50, 99, 130, 41, 102, 129, 20, 81, 69, 136, 65, 35, 89, 56, 66, 117, 91, 53, 131, 93, 58, 72, 3, 96, 10, 88, 114, 135, 115, 74, 29, 49, 138, 118, 36, 43, 75, 48, 8, 106, 73, 94, 70, 116, 84, 124, 80, 28, 13, 110, 120, 51, 113, 0, 59, 17, 25, 12, 119, 32, 15, 97, 23, 7, 107, 16, 52, 2, 30, 108, 14, 61, 92, 79, 18, 6, 90, 4, 63, 40, 33, 112, 83, 76, 57, 1, 122, 95, 47, 64, 111, 44, 55, 125, 137, 98, 46, 62, 77, 82, 19, 21, 5, 127, 68, 24, 132, 121, 78, 133, 34, 37, 9, 38, 103, 109, 134, 85, 42, 60, 45, 139, 39, 128, 100, 123, 87],
        [40, 56, 47, 118, 36, 73, 131, 20, 17, 28, 37, 128, 133, 26, 58, 98, 134, 139, 130, 23, 85, 119, 105, 68, 65, 42, 138, 51, 72, 34, 9, 88, 53, 123, 33, 4, 83, 63, 127, 99, 135, 132, 104, 74, 41, 27, 22, 77, 2, 46, 50, 66, 89, 35, 136, 30, 92, 101, 1, 129, 102, 5, 87, 21, 16, 62, 69, 76, 70, 113, 110, 15, 116, 32, 59, 115, 75, 91, 120, 125, 60, 121, 6, 49, 100, 14, 96, 82, 44, 67, 52, 24, 54, 90, 38, 57, 107, 112, 117, 109, 8, 0, 124, 43, 29, 103, 48, 95, 108, 84, 39, 106, 45, 31, 10, 64, 79, 80, 126, 86, 81, 7, 12, 78, 137, 3, 111, 11, 97, 18, 122, 94, 61, 25, 13, 55, 93, 71, 19, 114],
        [116, 3, 2, 127, 133, 64, 72, 80, 94, 41, 35, 84, 129, 27, 134, 91, 9, 100, 24, 54, 12, 59, 83, 31, 118, 114, 16, 22, 38, 58, 123, 44, 132, 25, 136, 124, 110, 67, 63, 20, 95, 19, 79, 69, 102, 23, 137, 56, 74, 128, 0, 117, 53, 29, 105, 50, 109, 85, 46, 101, 60, 70, 57, 10, 40, 78, 36, 90, 6, 88, 107, 49, 39, 92, 73, 99, 77, 13, 98, 103, 65, 76, 5, 93, 61, 11, 68, 4, 34, 104, 7, 51, 62, 97, 119, 17, 81, 52, 8, 33, 111, 86, 18, 47, 96, 115, 89, 28, 71, 14, 45, 75, 32, 30, 139, 21, 42, 48, 1, 66, 55, 37, 43, 125, 130, 112, 87, 121, 138, 120, 122, 106, 113, 26, 126, 131, 108, 135, 82, 15],
        [14, 38, 114, 100, 95, 12, 82, 26, 85, 115, 81, 4, 5, 58, 65, 139, 93, 78, 59, 135, 32, 45, 110, 92, 88, 64, 57, 40, 60, 50, 44, 8, 71, 90, 102, 133, 109, 47, 86, 106, 43, 62, 24, 113, 122, 130, 76, 112, 63, 46, 138, 97, 79, 9, 33, 37, 56, 23, 91, 101, 125, 119, 72, 20, 17, 118, 132, 108, 2, 16, 54, 11, 120, 25, 107, 69, 53, 6, 129, 124, 84, 89, 77, 30, 66, 83, 34, 21, 116, 87, 29, 123, 127, 117, 51, 68, 22, 1, 70, 19, 35, 96, 75, 98, 52, 10, 80, 7, 94, 134, 0, 3, 39, 104, 103, 61, 121, 13, 73, 55, 128, 74, 99, 126, 18, 137, 131, 136, 67, 111, 36, 42, 48, 105, 49, 15, 27, 31, 28, 41],
        [107, 80, 88, 71, 55, 29, 51, 10, 75, 124, 20, 56, 64, 38, 47, 28, 65, 6, 18, 90, 130, 119, 114, 37, 70, 103, 82, 99, 5, 101, 138, 54, 132, 111, 115, 34, 66, 85, 112, 87, 116, 12, 106, 98, 27, 83, 137, 7, 104, 84, 73, 61, 36, 58, 81, 60, 16, 31, 120, 53, 108, 63, 30, 105, 121, 3, 0, 32, 76, 79, 44, 129, 100, 40, 89, 11, 35, 113, 25, 48, 62, 86, 9, 23, 95, 24, 39, 14, 17, 57, 74, 1, 139, 110, 123, 109, 33, 19, 2, 102, 49, 135, 131, 59, 125, 72, 91, 15, 92, 13, 126, 97, 77, 46, 127, 122, 41, 136, 118, 52, 69, 117, 96, 26, 50, 133, 4, 42, 128, 21, 43, 68, 45, 78, 67, 22, 93, 134, 8, 94],
        [133, 44, 75, 95, 37, 41, 4, 6, 99, 77, 89, 80, 70, 88, 115, 97, 32, 100, 108, 64, 50, 29, 76, 98, 54, 131, 43, 83, 46, 19, 92, 102, 120, 135, 134, 31, 39, 47, 138, 69, 40, 119, 82, 58, 60, 53, 30, 34, 128, 13, 107, 45, 129, 91, 123, 27, 59, 103, 33, 15, 84, 20, 72, 81, 111, 24, 21, 86, 3, 114, 67, 9, 66, 22, 17, 137, 117, 122, 62, 74, 65, 48, 35, 28, 61, 55, 49, 85, 11, 18, 112, 57, 42, 106, 26, 96, 110, 109, 38, 118, 101, 113, 124, 126, 52, 14, 36, 8, 121, 104, 56, 94, 71, 73, 1, 79, 127, 87, 51, 90, 139, 132, 130, 68, 116, 78, 93, 7, 25, 16, 23, 136, 105, 5, 63, 12, 0, 125, 10, 2],
        [76, 107, 102, 109, 125, 92, 130, 138, 99, 103, 0, 71, 34, 93, 115, 13, 114, 89, 133, 4, 111, 98, 11, 56, 46, 94, 110, 87, 30, 126, 108, 39, 25, 123, 24, 121, 96, 50, 62, 19, 70, 43, 117, 72, 52, 16, 5, 80, 119, 105, 42, 97, 75, 7, 3, 64, 131, 22, 73, 61, 45, 26, 6, 116, 134, 112, 86, 67, 136, 40, 127, 23, 31, 44, 54, 74, 135, 18, 38, 104, 21, 68, 51, 124, 84, 77, 69, 55, 49, 60, 47, 95, 27, 29, 32, 85, 10, 28, 36, 83, 9, 81, 78, 20, 122, 106, 14, 63, 91, 90, 1, 33, 17, 41, 58, 113, 65, 53, 101, 35, 37, 79, 15, 8, 139, 2, 129, 57, 137, 88, 82, 12, 59, 132, 120, 48, 118, 100, 128, 66],
        [139, 122, 99, 81, 49, 113, 34, 52, 90, 11, 73, 120, 18, 79, 96, 135, 106, 59, 103, 36, 131, 57, 86, 125, 97, 22, 72, 56, 0, 58, 61, 4, 46, 77, 62, 121, 85, 123, 50, 102, 67, 31, 28, 80, 112, 136, 21, 104, 88, 114, 109, 30, 118, 41, 65, 117, 23, 111, 107, 53, 54, 13, 37, 95, 124, 7, 63, 137, 69, 74, 26, 89, 93, 38, 40, 51, 128, 2, 6, 78, 84, 27, 14, 94, 10, 42, 16, 115, 116, 126, 60, 66, 87, 68, 5, 130, 91, 127, 3, 9, 101, 71, 108, 45, 1, 20, 83, 33, 132, 24, 55, 43, 17, 39, 110, 119, 29, 133, 100, 48, 35, 105, 82, 98, 12, 8, 15, 47, 64, 76, 70, 129, 32, 138, 44, 19, 75, 25, 134, 92],
        [18, 36, 61, 126, 139, 42, 85, 98, 57, 37, 30, 6, 64, 131, 58, 108, 74, 15, 86, 29, 52, 51, 20, 22, 31, 133, 43, 21, 83, 118, 136, 70, 116, 25, 26, 45, 88, 97, 96, 48, 59, 77, 2, 55, 80, 62, 4, 128, 41, 84, 67, 110, 111, 23, 75, 91, 107, 5, 16, 14, 129, 92, 34, 101, 17, 7, 33, 9, 95, 8, 46, 90, 102, 56, 49, 123, 24, 3, 47, 119, 63, 115, 87, 78, 0, 120, 12, 38, 66, 121, 35, 32, 69, 132, 100, 60, 27, 82, 50, 40, 72, 109, 11, 13, 68, 44, 76, 93, 99, 53, 79, 124, 10, 28, 1, 71, 113, 19, 112, 105, 135, 137, 94, 103, 127, 89, 117, 106, 138, 125, 104, 73, 39, 114, 122, 54, 81, 65, 130, 134],
        [47, 125, 130, 138, 38, 72, 16, 122, 59, 77, 15, 101, 4, 102, 6, 93, 131, 56, 136, 85, 1, 127, 109, 133, 120, 17, 113, 49, 88, 98, 95, 91, 76, 61, 137, 111, 139, 2, 97, 35, 70, 25, 87, 128, 75, 48, 119, 22, 10, 36, 20, 19, 24, 99, 32, 0, 8, 9, 3, 43, 94, 86, 71, 50, 18, 80, 126, 117, 55, 23, 84, 65, 108, 58, 40, 74, 73, 41, 105, 51, 63, 30, 106, 54, 46, 115, 116, 26, 29, 82, 104, 118, 83, 132, 60, 107, 52, 31, 129, 69, 89, 90, 14, 53, 33, 67, 45, 92, 96, 103, 5, 114, 34, 42, 11, 81, 13, 64, 39, 135, 7, 78, 62, 21, 27, 57, 79, 112, 124, 37, 66, 110, 100, 123, 44, 121, 134, 68, 12, 28],
        [55, 10, 24, 138, 128, 86, 36, 14, 44, 62, 76, 118, 131, 61, 11, 27, 93, 116, 17, 0, 124, 72, 79, 28, 135, 112, 57, 7, 108, 120, 13, 22, 50, 15, 2, 51, 34, 40, 31, 109, 119, 132, 113, 115, 42, 92, 33, 123, 77, 74, 103, 111, 139, 130, 137, 94, 23, 3, 117, 134, 78, 12, 54, 114, 63, 82, 81, 133, 21, 106, 16, 58, 101, 45, 1, 30, 125, 60, 104, 96, 102, 83, 127, 70, 18, 73, 35, 88, 126, 47, 56, 121, 67, 29, 5, 99, 89, 107, 122, 37, 90, 85, 38, 110, 49, 129, 25, 100, 46, 20, 69, 91, 105, 32, 97, 136, 4, 71, 19, 53, 41, 66, 6, 87, 52, 98, 9, 26, 84, 64, 8, 80, 48, 65, 59, 68, 95, 39, 43, 75],
        [35, 56, 28, 75, 68, 106, 24, 9, 131, 66, 30, 85, 52, 89, 80, 25, 82, 47, 6, 40, 118, 34, 71, 67, 105, 113, 107, 96, 81, 21, 7, 94, 55, 22, 59, 33, 15, 101, 125, 5, 46, 93, 61, 17, 58, 87, 53, 115, 139, 111, 41, 29, 121, 64, 62, 97, 127, 126, 119, 0, 132, 74, 129, 13, 18, 73, 45, 8, 27, 70, 78, 44, 117, 109, 42, 63, 10, 76, 92, 37, 1, 84, 79, 135, 86, 2, 32, 31, 108, 77, 102, 50, 11, 99, 91, 136, 138, 20, 48, 116, 110, 122, 54, 88, 36, 12, 60, 23, 14, 100, 49, 19, 4, 133, 26, 130, 128, 98, 120, 3, 83, 134, 104, 123, 38, 137, 90, 51, 57, 103, 114, 16, 69, 72, 124, 43, 65, 95, 112, 39],
        [5, 6, 91, 3, 113, 48, 33, 44, 25, 81, 26, 94, 68, 85, 37, 83, 20, 77, 61, 112, 36, 47, 42, 125, 62, 92, 80, 89, 46, 132, 136, 14, 67, 104, 10, 103, 13, 79, 131, 64, 124, 93, 43, 51, 98, 127, 8, 2, 129, 7, 71, 135, 40, 82, 23, 49, 95, 123, 78, 29, 121, 122, 11, 24, 57, 1, 27, 9, 58, 34, 99, 137, 39, 4, 66, 109, 116, 35, 52, 50, 139, 65, 111, 134, 72, 106, 138, 54, 70, 133, 73, 88, 87, 38, 53, 96, 28, 16, 115, 63, 114, 100, 41, 45, 56, 22, 107, 32, 18, 19, 117, 76, 108, 55, 59, 110, 17, 120, 126, 21, 75, 97, 128, 60, 84, 102, 118, 0, 101, 15, 31, 69, 74, 105, 86, 30, 130, 90, 12, 119],
        [13, 126, 95, 9, 138, 71, 83, 108, 87, 33, 139, 46, 82, 11, 56, 99, 18, 0, 102, 90, 63, 107, 31, 112, 24, 23, 104, 88, 67, 27, 40, 8, 125, 15, 59, 123, 26, 44, 103, 116, 49, 55, 14, 97, 91, 38, 135, 51, 109, 115, 78, 80, 127, 130, 19, 111, 137, 76, 105, 119, 75, 28, 5, 122, 134, 77, 124, 120, 92, 47, 57, 12, 70, 68, 74, 32, 66, 131, 85, 7, 114, 136, 79, 132, 39, 45, 41, 81, 34, 60, 62, 22, 25, 72, 84, 118, 128, 10, 121, 50, 52, 133, 106, 96, 53, 129, 110, 93, 113, 101, 30, 29, 48, 16, 69, 61, 100, 86, 3, 21, 42, 98, 65, 73, 20, 6, 4, 43, 17, 35, 36, 89, 117, 37, 58, 1, 64, 2, 54, 94],
        [10, 48, 6, 24, 17, 47, 84, 78, 37, 35, 49, 130, 11, 136, 67, 99, 100, 30, 69, 64, 65, 41, 72, 114, 44, 81, 15, 23, 75, 87, 82, 85, 59, 7, 27, 63, 125, 92, 21, 14, 119, 54, 33, 20, 109, 2, 134, 123, 111, 126, 104, 46, 124, 132, 71, 128, 25, 31, 120, 101, 116, 127, 58, 28, 8, 13, 135, 121, 90, 36, 51, 138, 16, 68, 91, 103, 106, 9, 108, 95, 42, 38, 113, 73, 3, 32, 43, 60, 98, 1, 115, 12, 117, 19, 5, 50, 55, 18, 105, 129, 77, 89, 107, 76, 61, 57, 40, 29, 112, 62, 52, 96, 74, 118, 83, 66, 137, 56, 139, 97, 34, 70, 93, 110, 79, 45, 133, 22, 39, 102, 53, 94, 4, 86, 131, 0, 88, 26, 122, 80],
        [95, 93, 137, 19, 31, 125, 89, 0, 23, 118, 102, 29, 121, 135, 72, 101, 49, 132, 62, 35, 138, 120, 94, 134, 73, 32, 86, 56, 128, 16, 28, 63, 45, 71, 58, 11, 131, 99, 20, 129, 15, 82, 7, 44, 40, 107, 97, 75, 41, 3, 111, 46, 114, 78, 34, 48, 25, 65, 24, 130, 2, 74, 42, 106, 92, 61, 79, 55, 122, 64, 133, 69, 139, 27, 112, 12, 104, 66, 37, 39, 4, 21, 77, 5, 8, 123, 109, 127, 76, 81, 51, 13, 52, 9, 50, 14, 70, 18, 43, 6, 59, 54, 100, 136, 36, 88, 116, 105, 96, 30, 84, 10, 90, 87, 115, 85, 91, 22, 26, 47, 119, 17, 68, 57, 1, 80, 60, 103, 117, 98, 110, 53, 67, 126, 38, 113, 33, 108, 83, 124],
        [4, 131, 67, 46, 76, 130, 47, 19, 45, 14, 21, 98, 93, 111, 134, 112, 23, 17, 78, 63, 64, 129, 65, 124, 69, 30, 115, 125, 85, 95, 106, 15, 68, 1, 99, 44, 49, 61, 132, 133, 89, 57, 35, 24, 91, 51, 139, 96, 5, 66, 58, 2, 9, 101, 38, 127, 72, 79, 119, 107, 41, 135, 138, 97, 116, 0, 6, 10, 31, 52, 77, 33, 59, 62, 22, 13, 36, 103, 60, 83, 12, 71, 86, 25, 90, 55, 28, 20, 32, 114, 16, 136, 120, 102, 87, 8, 82, 29, 80, 104, 7, 108, 105, 73, 109, 118, 27, 81, 34, 92, 113, 122, 37, 137, 11, 18, 110, 50, 39, 26, 40, 43, 53, 123, 54, 75, 117, 70, 94, 128, 74, 48, 121, 88, 126, 42, 100, 3, 84, 56],
        [131, 90, 36, 72, 67, 136, 97, 83, 9, 7, 24, 101, 38, 51, 79, 117, 62, 52, 8, 106, 116, 112, 27, 127, 133, 54, 26, 107, 99, 138, 0, 46, 14, 12, 19, 5, 130, 66, 87, 129, 68, 122, 139, 71, 95, 105, 74, 10, 25, 11, 128, 118, 31, 59, 56, 120, 45, 96, 65, 82, 92, 77, 64, 18, 126, 60, 78, 55, 103, 93, 39, 58, 49, 115, 108, 113, 47, 33, 75, 13, 88, 61, 57, 1, 110, 23, 98, 81, 32, 20, 114, 85, 48, 102, 76, 84, 70, 29, 121, 21, 100, 104, 35, 30, 50, 15, 89, 2, 41, 73, 3, 37, 28, 123, 22, 135, 137, 53, 43, 124, 6, 86, 109, 17, 4, 42, 134, 69, 91, 132, 119, 111, 94, 44, 63, 125, 16, 80, 34, 40],
        [43, 32, 59, 56, 20, 22, 75, 101, 78, 117, 27, 0, 72, 70, 25, 105, 136, 16, 113, 115, 89, 88, 57, 138, 61, 107, 2, 14, 84, 64, 10, 98, 111, 92, 132, 108, 100, 96, 85, 126, 83, 130, 131, 106, 79, 81, 71, 50, 6, 91, 82, 99, 124, 18, 5, 52, 17, 51, 73, 45, 44, 28, 67, 7, 54, 37, 103, 19, 23, 55, 139, 38, 69, 112, 66, 97, 36, 8, 60, 15, 80, 29, 58, 68, 74, 122, 114, 127, 110, 40, 46, 34, 118, 94, 134, 26, 102, 133, 116, 42, 109, 128, 21, 87, 4, 104, 95, 62, 125, 86, 41, 120, 35, 53, 31, 49, 119, 9, 123, 11, 77, 135, 93, 65, 47, 30, 90, 39, 129, 121, 24, 48, 76, 33, 1, 3, 13, 12, 137, 63],
        [37, 63, 64, 123, 55, 72, 52, 18, 5, 99, 79, 110, 49, 135, 45, 124, 131, 77, 85, 4, 68, 82, 8, 58, 51, 30, 33, 96, 87, 43, 126, 25, 118, 13, 34, 11, 130, 137, 40, 119, 80, 98, 106, 27, 104, 62, 132, 103, 120, 93, 0, 47, 6, 127, 53, 139, 50, 97, 71, 19, 113, 70, 102, 1, 133, 128, 129, 84, 78, 75, 39, 48, 112, 7, 28, 59, 115, 136, 23, 92, 116, 69, 73, 14, 138, 21, 57, 111, 36, 42, 91, 74, 3, 76, 46, 94, 22, 12, 125, 15, 10, 60, 100, 16, 66, 32, 121, 2, 89, 107, 134, 56, 109, 29, 31, 81, 108, 61, 90, 9, 83, 117, 41, 122, 88, 26, 95, 35, 54, 105, 67, 114, 20, 65, 17, 86, 101, 38, 44, 24],
        [7, 77, 132, 28, 36, 58, 48, 96, 19, 137, 47, 91, 22, 114, 117, 127, 46, 108, 67, 54, 49, 11, 107, 105, 71, 31, 135, 18, 35, 81, 5, 119, 59, 63, 116, 17, 44, 95, 14, 88, 20, 134, 23, 102, 12, 133, 1, 94, 39, 43, 109, 123, 66, 69, 115, 25, 121, 24, 41, 86, 113, 136, 98, 45, 30, 97, 126, 51, 0, 75, 104, 65, 118, 64, 100, 6, 32, 131, 130, 78, 34, 112, 29, 70, 101, 93, 79, 124, 111, 85, 73, 40, 62, 13, 26, 15, 10, 99, 60, 76, 9, 87, 80, 92, 8, 27, 138, 16, 125, 57, 120, 33, 68, 129, 89, 90, 139, 55, 72, 110, 42, 37, 56, 21, 50, 61, 4, 122, 2, 38, 84, 53, 3, 52, 106, 103, 83, 74, 82, 128],
        [78, 106, 104, 84, 127, 105, 120, 129, 29, 54, 91, 135, 100, 62, 125, 90, 4, 130, 0, 44, 6, 9, 111, 93, 128, 114, 58, 79, 33, 121, 69, 122, 52, 124, 98, 67, 63, 95, 139, 131, 88, 83, 28, 101, 97, 117, 40, 118, 60, 99, 34, 75, 43, 31, 21, 82, 27, 57, 11, 48, 86, 138, 7, 35, 61, 17, 81, 113, 50, 73, 56, 10, 22, 92, 25, 89, 26, 18, 102, 66, 123, 108, 109, 94, 119, 96, 30, 53, 74, 5, 47, 14, 38, 80, 59, 110, 71, 19, 36, 77, 12, 8, 112, 68, 72, 45, 134, 15, 126, 76, 42, 55, 24, 23, 64, 51, 116, 20, 87, 132, 115, 41, 13, 32, 49, 46, 107, 133, 2, 1, 136, 3, 103, 85, 37, 70, 16, 39, 65, 137],
        [96, 29, 45, 105, 107, 99, 35, 4, 62, 127, 82, 42, 138, 74, 121, 53, 27, 83, 48, 13, 18, 78, 40, 87, 19, 65, 129, 28, 106, 61, 34, 33, 133, 51, 14, 3, 66, 125, 1, 118, 55, 80, 86, 20, 23, 76, 132, 37, 72, 21, 122, 92, 44, 11, 2, 49, 60, 115, 38, 54, 70, 137, 36, 12, 16, 134, 57, 64, 25, 114, 124, 68, 84, 17, 46, 94, 41, 89, 108, 116, 139, 93, 88, 97, 9, 131, 26, 22, 98, 120, 5, 109, 117, 52, 56, 111, 79, 81, 8, 7, 112, 63, 73, 15, 102, 113, 128, 50, 71, 77, 101, 90, 95, 126, 67, 110, 69, 119, 10, 136, 59, 91, 39, 75, 0, 130, 100, 123, 85, 103, 24, 43, 135, 104, 30, 31, 47, 6, 58, 32],
        [47, 10, 105, 56, 75, 45, 134, 101, 25, 87, 115, 42, 135, 79, 20, 66, 9, 39, 116, 82, 73, 131, 83, 90, 64, 51, 57, 46, 86, 110, 107, 28, 30, 113, 59, 49, 13, 88, 8, 103, 54, 33, 92, 104, 114, 125, 34, 77, 67, 128, 16, 15, 19, 121, 50, 89, 38, 21, 123, 23, 137, 41, 119, 133, 97, 136, 117, 58, 122, 44, 4, 65, 35, 102, 126, 124, 139, 120, 61, 22, 106, 5, 95, 72, 2, 26, 43, 118, 62, 78, 74, 108, 109, 69, 111, 132, 81, 55, 3, 99, 37, 80, 85, 32, 112, 138, 130, 93, 52, 7, 94, 84, 40, 18, 70, 91, 63, 129, 98, 68, 12, 76, 31, 24, 53, 60, 48, 0, 11, 71, 14, 29, 36, 127, 100, 6, 27, 17, 1, 96],
        [35, 124, 29, 7, 84, 66, 20, 83, 99, 130, 0, 9, 46, 55, 133, 19, 71, 2, 111, 88, 106, 90, 39, 132, 68, 73, 107, 70, 75, 28, 122, 101, 15, 48, 14, 57, 117, 123, 92, 40, 103, 26, 138, 119, 36, 72, 24, 95, 1, 4, 47, 89, 81, 139, 105, 137, 112, 25, 11, 108, 56, 97, 127, 69, 34, 82, 120, 110, 121, 42, 18, 116, 23, 16, 93, 8, 37, 58, 53, 12, 96, 134, 100, 60, 43, 67, 54, 13, 30, 51, 131, 64, 91, 78, 76, 113, 45, 21, 32, 94, 59, 118, 49, 63, 27, 38, 3, 41, 61, 102, 128, 6, 65, 31, 115, 86, 74, 5, 17, 33, 50, 10, 135, 62, 125, 44, 136, 98, 85, 129, 80, 114, 87, 22, 52, 104, 109, 126, 79, 77],
        [83, 47, 125, 91, 92, 8, 25, 6, 77, 119, 109, 39, 15, 67, 107, 22, 75, 113, 110, 31, 53, 103, 124, 84, 54, 20, 134, 116, 94, 102, 87, 123, 93, 56, 136, 99, 30, 19, 115, 52, 105, 122, 139, 73, 133, 24, 0, 82, 50, 62, 118, 45, 51, 106, 79, 5, 12, 127, 32, 68, 16, 69, 97, 55, 100, 57, 23, 11, 85, 90, 35, 58, 96, 60, 26, 29, 36, 49, 64, 95, 137, 117, 72, 38, 43, 132, 21, 78, 98, 88, 44, 108, 28, 63, 81, 61, 7, 70, 76, 129, 9, 130, 48, 14, 27, 86, 37, 42, 121, 89, 1, 101, 66, 126, 65, 111, 3, 10, 2, 46, 131, 33, 34, 104, 114, 74, 59, 138, 120, 80, 17, 4, 112, 135, 128, 40, 18, 41, 13, 71],
        [87, 57, 27, 33, 74, 91, 49, 80, 127, 68, 88, 93, 48, 14, 99, 51, 58, 73, 18, 16, 116, 122, 19, 13, 10, 54, 4, 119, 101, 109, 60, 108, 53, 123, 76, 44, 125, 117, 36, 29, 98, 115, 124, 77, 12, 135, 64, 11, 55, 40, 2, 132, 128, 75, 105, 9, 34, 50, 39, 0, 30, 90, 47, 84, 59, 28, 3, 66, 72, 86, 15, 131, 134, 20, 133, 32, 94, 92, 139, 69, 8, 79, 136, 114, 70, 138, 43, 96, 83, 106, 112, 1, 35, 42, 137, 6, 111, 17, 21, 110, 85, 7, 121, 113, 61, 46, 120, 89, 37, 97, 129, 71, 103, 22, 38, 126, 45, 104, 100, 31, 107, 23, 65, 130, 24, 41, 25, 56, 78, 52, 62, 95, 67, 82, 118, 102, 26, 5, 63, 81],
        [131, 63, 96, 120, 34, 139, 27, 118, 60, 72, 122, 73, 44, 49, 100, 111, 87, 93, 85, 5, 31, 24, 55, 32, 21, 129, 134, 75, 82, 51, 104, 53, 26, 119, 52, 113, 19, 22, 102, 74, 39, 67, 36, 47, 33, 101, 25, 3, 110, 18, 76, 133, 126, 116, 56, 11, 0, 16, 46, 13, 108, 41, 66, 94, 114, 128, 10, 29, 57, 97, 137, 54, 37, 70, 61, 8, 7, 90, 71, 105, 48, 59, 12, 99, 121, 86, 14, 78, 77, 43, 30, 68, 95, 42, 9, 28, 115, 123, 117, 69, 88, 109, 58, 98, 4, 107, 62, 89, 91, 84, 124, 6, 2, 83, 81, 40, 79, 106, 35, 103, 15, 23, 17, 130, 38, 136, 50, 64, 135, 127, 125, 20, 132, 45, 92, 65, 112, 1, 138, 80],
        [68, 29, 134, 37, 4, 106, 42, 13, 50, 0, 19, 26, 112, 79, 80, 136, 36, 138, 49, 85, 122, 82, 109, 57, 69, 48, 73, 66, 120, 31, 71, 90, 118, 139, 100, 35, 1, 84, 86, 38, 67, 10, 55, 132, 137, 74, 16, 8, 102, 115, 11, 25, 81, 114, 76, 83, 20, 107, 131, 113, 97, 5, 39, 64, 6, 130, 60, 77, 52, 126, 128, 111, 103, 124, 43, 44, 104, 96, 63, 91, 92, 40, 28, 110, 47, 2, 21, 117, 7, 94, 14, 46, 119, 30, 59, 116, 127, 23, 65, 72, 45, 105, 58, 22, 89, 125, 108, 129, 135, 62, 70, 34, 95, 88, 78, 33, 61, 98, 123, 24, 75, 121, 27, 133, 3, 101, 51, 54, 53, 41, 12, 87, 18, 99, 17, 15, 93, 9, 56, 32],
        [124, 64, 109, 42, 105, 78, 134, 77, 9, 92, 87, 8, 69, 114, 17, 81, 91, 120, 32, 84, 35, 41, 12, 98, 61, 19, 5, 86, 6, 40, 95, 89, 15, 110, 132, 113, 94, 26, 121, 85, 33, 3, 63, 88, 2, 44, 14, 75, 119, 76, 45, 47, 96, 1, 106, 10, 128, 39, 137, 11, 108, 30, 68, 54, 18, 16, 72, 123, 51, 31, 7, 37, 23, 112, 59, 131, 27, 52, 133, 71, 65, 139, 49, 43, 73, 58, 34, 53, 74, 115, 117, 83, 97, 36, 122, 116, 4, 56, 130, 66, 118, 70, 102, 111, 93, 101, 57, 46, 127, 60, 29, 38, 62, 82, 135, 79, 0, 125, 100, 103, 138, 136, 48, 22, 107, 28, 21, 25, 24, 80, 55, 20, 67, 129, 50, 90, 99, 104, 13, 126],
        [17, 84, 95, 24, 73, 3, 93, 30, 9, 108, 11, 127, 135, 32, 19, 78, 2, 61, 7, 130, 33, 26, 64, 94, 112, 43, 6, 40, 63, 131, 47, 69, 97, 4, 57, 80, 12, 20, 50, 42, 126, 107, 53, 35, 83, 16, 67, 70, 82, 51, 15, 22, 120, 113, 44, 46, 125, 116, 122, 77, 0, 76, 81, 71, 101, 38, 41, 79, 28, 137, 121, 128, 118, 134, 45, 124, 74, 36, 29, 90, 85, 8, 88, 68, 54, 34, 114, 59, 23, 48, 62, 55, 72, 111, 129, 139, 91, 13, 103, 52, 89, 1, 86, 58, 119, 49, 87, 133, 136, 106, 99, 110, 27, 98, 123, 75, 104, 31, 92, 25, 18, 105, 5, 117, 100, 10, 14, 56, 102, 138, 66, 132, 115, 39, 96, 21, 37, 60, 65, 109],
        [99, 90, 71, 1, 51, 4, 64, 131, 58, 8, 112, 101, 10, 7, 5, 119, 52, 61, 46, 79, 96, 32, 98, 113, 39, 132, 137, 92, 120, 122, 121, 117, 41, 93, 73, 44, 123, 78, 53, 15, 34, 108, 29, 20, 54, 135, 43, 138, 129, 3, 76, 0, 103, 50, 25, 47, 45, 133, 110, 18, 115, 28, 72, 74, 9, 95, 56, 102, 80, 14, 130, 94, 81, 107, 118, 59, 104, 2, 105, 134, 33, 26, 109, 70, 60, 17, 42, 11, 62, 91, 97, 24, 66, 116, 83, 87, 69, 27, 114, 111, 13, 84, 36, 100, 38, 19, 139, 106, 22, 12, 49, 55, 67, 127, 89, 35, 31, 126, 65, 86, 124, 68, 21, 23, 37, 82, 85, 57, 30, 40, 128, 6, 16, 75, 125, 63, 77, 136, 48, 88],
        [7, 42, 3, 30, 83, 125, 119, 67, 0, 66, 93, 114, 14, 131, 40, 32, 44, 74, 34, 65, 116, 123, 126, 17, 102, 18, 2, 133, 37, 100, 71, 60, 112, 13, 103, 48, 23, 98, 70, 124, 110, 24, 62, 59, 33, 128, 76, 4, 120, 58, 78, 57, 107, 75, 68, 31, 117, 82, 130, 138, 118, 80, 16, 52, 43, 50, 122, 77, 94, 54, 1, 85, 47, 134, 12, 49, 69, 99, 97, 86, 72, 21, 61, 73, 96, 105, 109, 95, 139, 10, 15, 91, 19, 115, 9, 108, 5, 51, 20, 63, 127, 101, 64, 113, 55, 121, 6, 79, 22, 38, 28, 45, 81, 25, 132, 35, 135, 39, 88, 129, 92, 11, 137, 106, 26, 36, 53, 111, 29, 56, 89, 87, 41, 90, 104, 46, 84, 8, 136, 27],
        [0, 85, 53, 66, 96, 40, 3, 100, 25, 120, 71, 109, 28, 106, 45, 91, 126, 136, 123, 16, 75, 54, 80, 119, 7, 68, 128, 129, 38, 13, 22, 59, 139, 135, 82, 87, 41, 26, 99, 105, 138, 43, 77, 137, 133, 92, 121, 17, 1, 33, 60, 113, 46, 130, 107, 93, 15, 19, 81, 8, 116, 118, 72, 24, 48, 76, 95, 31, 18, 114, 34, 65, 97, 9, 83, 39, 70, 102, 131, 30, 94, 111, 2, 37, 112, 88, 74, 5, 127, 110, 36, 52, 108, 73, 56, 11, 42, 51, 104, 4, 21, 47, 62, 125, 55, 10, 32, 101, 27, 6, 78, 67, 14, 86, 98, 44, 61, 49, 132, 90, 64, 122, 58, 29, 12, 35, 124, 23, 103, 57, 115, 84, 20, 50, 63, 69, 134, 117, 79, 89],
        [134, 26, 127, 53, 12, 131, 107, 7, 9, 120, 108, 89, 106, 25, 37, 16, 84, 125, 101, 95, 62, 48, 79, 82, 15, 123, 63, 61, 110, 52, 13, 41, 90, 49, 2, 105, 21, 80, 72, 135, 65, 99, 29, 60, 137, 128, 77, 30, 116, 64, 78, 93, 43, 31, 73, 94, 118, 81, 115, 111, 88, 1, 136, 36, 119, 20, 5, 133, 102, 117, 76, 32, 97, 0, 18, 47, 35, 14, 46, 3, 66, 87, 91, 68, 22, 40, 38, 122, 74, 6, 121, 109, 112, 42, 124, 39, 27, 11, 67, 92, 86, 103, 132, 28, 56, 70, 126, 130, 58, 4, 34, 71, 96, 17, 19, 45, 139, 44, 33, 85, 138, 24, 75, 50, 51, 23, 83, 55, 10, 129, 69, 100, 59, 113, 114, 8, 54, 98, 57, 104],
        [136, 139, 65, 114, 70, 130, 22, 20, 84, 25, 60, 107, 108, 52, 69, 0, 138, 29, 104, 46, 101, 3, 119, 58, 17, 125, 44, 61, 98, 41, 12, 67, 33, 134, 129, 121, 7, 8, 59, 77, 31, 47, 81, 99, 124, 94, 91, 135, 42, 27, 30, 51, 50, 21, 90, 11, 96, 19, 16, 43, 103, 55, 92, 73, 54, 32, 6, 131, 15, 53, 120, 14, 111, 102, 34, 86, 28, 112, 9, 5, 72, 40, 113, 105, 132, 4, 122, 110, 62, 89, 128, 48, 133, 49, 68, 64, 115, 45, 78, 106, 57, 26, 39, 18, 71, 74, 38, 76, 75, 35, 117, 100, 80, 56, 83, 137, 36, 85, 79, 123, 23, 95, 97, 13, 118, 116, 10, 1, 109, 82, 66, 24, 2, 127, 126, 37, 63, 87, 93, 88],
        [139, 94, 34, 91, 43, 2, 9, 88, 65, 132, 120, 128, 7, 59, 110, 52, 93, 32, 73, 29, 74, 51, 136, 39, 36, 0, 131, 97, 62, 31, 123, 81, 42, 38, 83, 78, 37, 103, 127, 11, 30, 98, 86, 102, 40, 111, 118, 70, 125, 46, 122, 14, 56, 75, 67, 64, 126, 101, 22, 96, 72, 15, 95, 90, 19, 137, 49, 99, 60, 54, 129, 105, 18, 16, 124, 84, 8, 61, 50, 33, 45, 24, 53, 6, 44, 69, 130, 48, 47, 134, 104, 77, 13, 79, 23, 138, 85, 121, 92, 17, 109, 25, 76, 26, 119, 135, 20, 5, 68, 87, 107, 27, 12, 71, 114, 3, 4, 58, 108, 10, 21, 113, 115, 28, 66, 100, 1, 57, 41, 116, 35, 112, 117, 133, 89, 63, 82, 106, 55, 80],
        [37, 86, 93, 137, 110, 30, 85, 105, 100, 11, 8, 57, 126, 28, 117, 73, 0, 43, 136, 51, 21, 91, 24, 66, 25, 38, 96, 90, 39, 89, 107, 33, 14, 139, 123, 131, 26, 23, 59, 34, 13, 60, 116, 130, 101, 70, 3, 62, 135, 54, 134, 82, 31, 64, 138, 6, 49, 18, 78, 109, 67, 19, 92, 98, 102, 74, 79, 7, 56, 65, 95, 52, 111, 97, 53, 45, 40, 29, 121, 71, 133, 41, 17, 132, 32, 46, 118, 50, 99, 119, 106, 114, 88, 75, 1, 2, 22, 69, 94, 35, 113, 5, 83, 61, 9, 124, 36, 27, 120, 125, 63, 115, 84, 10, 104, 77, 87, 76, 12, 48, 122, 68, 128, 81, 20, 42, 129, 44, 16, 47, 108, 72, 15, 4, 55, 58, 103, 127, 112, 80],
        [93, 9, 48, 49, 7, 38, 51, 22, 13, 109, 5, 67, 58, 55, 70, 21, 105, 111, 63, 25, 118, 113, 130, 122, 18, 102, 115, 132, 76, 71, 84, 39, 120, 129, 35, 29, 106, 10, 65, 124, 45, 92, 139, 60, 64, 52, 107, 134, 97, 86, 62, 103, 19, 73, 114, 24, 1, 123, 81, 12, 88, 28, 31, 34, 53, 59, 87, 137, 136, 30, 127, 82, 26, 95, 128, 32, 89, 6, 131, 2, 74, 15, 43, 100, 135, 0, 54, 96, 133, 16, 69, 119, 40, 20, 50, 33, 23, 121, 138, 112, 72, 79, 116, 98, 125, 8, 126, 27, 94, 41, 75, 17, 99, 42, 91, 77, 66, 57, 104, 85, 11, 110, 46, 68, 56, 80, 37, 117, 83, 44, 4, 47, 90, 61, 14, 3, 36, 108, 101, 78],
        [104, 72, 21, 33, 99, 122, 106, 114, 136, 60, 32, 50, 16, 53, 39, 20, 138, 91, 59, 3, 113, 67, 66, 137, 77, 36, 108, 75, 95, 83, 85, 98, 55, 10, 15, 110, 101, 45, 14, 62, 126, 118, 107, 81, 86, 71, 30, 73, 87, 96, 48, 11, 115, 47, 9, 5, 1, 56, 34, 61, 121, 28, 89, 80, 42, 100, 70, 78, 93, 117, 112, 97, 129, 49, 12, 82, 25, 130, 52, 58, 63, 40, 116, 17, 57, 54, 111, 27, 79, 38, 103, 94, 8, 6, 133, 84, 109, 35, 92, 64, 131, 105, 76, 23, 139, 13, 46, 120, 44, 41, 19, 74, 132, 127, 102, 134, 88, 24, 123, 68, 7, 37, 135, 51, 2, 119, 69, 29, 18, 0, 65, 22, 124, 26, 125, 90, 31, 43, 128, 4],
        [86, 119, 70, 91, 128, 15, 129, 55, 76, 133, 25, 106, 19, 134, 29, 101, 14, 74, 95, 115, 90, 122, 9, 135, 16, 39, 109, 108, 103, 65, 130, 137, 114, 68, 21, 38, 33, 88, 59, 85, 41, 1, 7, 3, 4, 35, 77, 50, 5, 102, 64, 10, 113, 84, 123, 12, 125, 139, 61, 57, 94, 99, 30, 23, 81, 49, 6, 8, 54, 100, 92, 31, 116, 0, 46, 78, 105, 98, 2, 44, 42, 131, 93, 67, 87, 121, 66, 126, 73, 120, 43, 22, 127, 124, 79, 27, 53, 80, 26, 52, 75, 24, 37, 117, 18, 112, 62, 17, 104, 47, 20, 63, 71, 36, 110, 138, 51, 13, 83, 45, 136, 132, 32, 107, 11, 97, 89, 118, 56, 58, 40, 69, 111, 96, 48, 82, 28, 60, 34, 72],
        [29, 15, 101, 78, 37, 110, 20, 10, 109, 113, 75, 125, 21, 72, 89, 116, 12, 52, 70, 117, 42, 128, 16, 53, 19, 65, 18, 93, 26, 38, 58, 100, 11, 79, 14, 46, 74, 126, 103, 71, 133, 132, 129, 94, 40, 23, 97, 111, 61, 84, 27, 80, 3, 92, 44, 82, 45, 90, 48, 136, 25, 49, 118, 31, 2, 131, 120, 77, 57, 33, 32, 86, 134, 47, 83, 63, 59, 139, 107, 30, 6, 135, 69, 138, 41, 54, 122, 7, 81, 36, 22, 88, 4, 87, 95, 13, 104, 0, 114, 91, 62, 105, 137, 39, 112, 99, 43, 123, 9, 119, 34, 1, 106, 51, 76, 73, 66, 67, 56, 17, 60, 102, 130, 5, 50, 124, 24, 121, 98, 85, 68, 35, 28, 108, 8, 115, 96, 55, 127, 64],
        [28, 74, 86, 117, 138, 19, 108, 68, 91, 139, 47, 5, 120, 53, 18, 97, 112, 100, 127, 84, 87, 132, 71, 4, 20, 62, 46, 82, 49, 126, 37, 128, 94, 13, 125, 80, 3, 25, 96, 1, 38, 98, 31, 51, 69, 26, 99, 124, 43, 103, 90, 72, 89, 17, 81, 10, 130, 135, 93, 12, 39, 78, 119, 29, 110, 118, 11, 101, 85, 36, 83, 30, 67, 54, 104, 8, 64, 88, 14, 102, 63, 77, 23, 22, 58, 73, 106, 134, 59, 79, 123, 105, 115, 60, 42, 121, 0, 95, 136, 48, 7, 116, 33, 70, 50, 107, 56, 76, 6, 109, 111, 113, 66, 35, 27, 34, 114, 16, 65, 75, 131, 137, 61, 9, 52, 129, 15, 41, 44, 133, 57, 55, 122, 92, 45, 24, 40, 21, 2, 32],
        [97, 58, 132, 54, 27, 43, 138, 139, 73, 32, 40, 120, 89, 52, 39, 75, 49, 20, 9, 3, 71, 81, 95, 8, 0, 109, 128, 123, 18, 112, 116, 64, 56, 87, 84, 107, 80, 94, 22, 16, 124, 13, 65, 131, 26, 14, 2, 50, 57, 118, 113, 126, 7, 83, 51, 12, 98, 67, 61, 91, 78, 25, 82, 100, 31, 28, 76, 33, 77, 34, 29, 130, 42, 15, 92, 90, 102, 45, 121, 93, 110, 4, 99, 62, 38, 119, 117, 134, 70, 17, 127, 46, 122, 96, 86, 111, 53, 30, 44, 47, 125, 1, 114, 79, 72, 55, 66, 60, 59, 48, 37, 74, 68, 19, 23, 104, 105, 69, 24, 41, 129, 101, 133, 135, 85, 88, 106, 137, 35, 136, 36, 21, 11, 5, 63, 108, 10, 115, 6, 103],
        [41, 104, 85, 75, 90, 131, 25, 135, 133, 105, 125, 28, 80, 8, 9, 112, 97, 13, 3, 42, 100, 4, 113, 20, 74, 96, 73, 21, 134, 37, 130, 69, 52, 59, 79, 45, 5, 115, 40, 117, 24, 54, 56, 62, 139, 98, 57, 84, 10, 36, 38, 64, 114, 91, 102, 111, 16, 11, 128, 46, 89, 23, 103, 53, 29, 88, 39, 136, 124, 49, 83, 86, 12, 27, 34, 19, 55, 107, 44, 93, 1, 109, 0, 126, 137, 14, 43, 127, 108, 116, 78, 22, 132, 123, 76, 6, 26, 58, 33, 63, 70, 31, 68, 94, 110, 61, 121, 119, 138, 30, 99, 129, 81, 122, 48, 32, 65, 35, 67, 7, 51, 66, 15, 87, 47, 82, 17, 77, 2, 92, 120, 95, 106, 72, 118, 101, 18, 50, 60, 71],
        [16, 30, 118, 84, 79, 55, 108, 116, 67, 20, 2, 124, 54, 25, 117, 28, 72, 17, 70, 135, 90, 112, 71, 48, 114, 31, 1, 12, 136, 69, 22, 138, 128, 129, 123, 47, 74, 58, 99, 24, 18, 91, 35, 15, 77, 95, 115, 43, 9, 130, 107, 6, 7, 81, 5, 121, 53, 93, 86, 8, 68, 73, 45, 109, 59, 103, 19, 100, 21, 32, 102, 60, 52, 111, 4, 13, 66, 40, 137, 88, 87, 85, 56, 110, 36, 64, 65, 101, 57, 50, 76, 61, 94, 125, 132, 23, 26, 38, 126, 96, 63, 0, 83, 106, 134, 127, 97, 42, 104, 131, 105, 29, 33, 27, 120, 80, 3, 98, 41, 10, 78, 39, 34, 82, 119, 11, 139, 49, 62, 89, 37, 122, 46, 14, 92, 75, 113, 44, 51, 133],
        [65, 105, 106, 94, 119, 70, 41, 99, 35, 134, 32, 46, 2, 126, 112, 6, 81, 29, 92, 96, 11, 10, 116, 130, 61, 123, 115, 113, 87, 42, 69, 16, 44, 20, 132, 68, 14, 128, 60, 90, 21, 8, 39, 125, 104, 79, 34, 24, 4, 84, 124, 17, 103, 12, 93, 110, 66, 18, 129, 111, 78, 108, 23, 31, 122, 9, 83, 48, 33, 118, 7, 120, 107, 133, 56, 25, 52, 43, 73, 58, 62, 30, 91, 22, 0, 82, 40, 102, 97, 27, 85, 98, 49, 139, 19, 36, 117, 74, 28, 45, 57, 64, 54, 72, 13, 80, 59, 86, 89, 101, 121, 135, 88, 26, 76, 114, 137, 38, 50, 127, 71, 15, 75, 95, 131, 138, 5, 51, 53, 1, 37, 55, 3, 109, 63, 67, 136, 47, 77, 100],
        [135, 48, 21, 15, 69, 122, 106, 139, 7, 74, 99, 26, 129, 103, 102, 63, 93, 118, 111, 87, 9, 70, 51, 4, 34, 47, 114, 127, 138, 24, 12, 86, 49, 94, 38, 80, 76, 126, 13, 115, 29, 113, 27, 40, 42, 45, 59, 104, 3, 66, 1, 11, 134, 121, 25, 85, 109, 55, 96, 68, 108, 132, 105, 36, 33, 131, 58, 100, 28, 83, 41, 120, 84, 89, 30, 35, 128, 112, 136, 44, 62, 91, 22, 88, 67, 90, 60, 17, 39, 133, 65, 97, 77, 107, 37, 0, 110, 64, 116, 137, 57, 14, 19, 124, 31, 16, 5, 23, 75, 123, 18, 10, 71, 81, 130, 72, 52, 53, 20, 119, 73, 54, 61, 8, 6, 32, 117, 2, 78, 50, 125, 82, 95, 79, 46, 56, 43, 92, 98, 101],
        [106, 18, 20, 64, 117, 91, 133, 22, 34, 78, 54, 93, 24, 30, 73, 109, 112, 139, 70, 113, 79, 77, 25, 84, 81, 44, 115, 14, 90, 33, 104, 108, 85, 118, 99, 16, 125, 26, 29, 42, 82, 8, 68, 88, 60, 10, 137, 132, 56, 13, 2, 116, 83, 101, 134, 80, 49, 103, 51, 98, 15, 107, 65, 27, 46, 74, 62, 131, 135, 50, 58, 17, 128, 40, 12, 130, 71, 136, 48, 67, 94, 35, 97, 129, 72, 7, 102, 57, 3, 38, 61, 11, 4, 55, 126, 19, 39, 110, 111, 52, 121, 53, 45, 21, 114, 76, 47, 96, 87, 6, 89, 119, 23, 120, 43, 105, 63, 9, 69, 59, 32, 100, 75, 0, 123, 36, 28, 127, 41, 92, 138, 31, 37, 66, 5, 95, 122, 124, 1, 86],
        [49, 52, 26, 35, 81, 104, 85, 15, 75, 80, 0, 84, 91, 41, 110, 62, 77, 89, 113, 54, 31, 82, 44, 3, 122, 8, 67, 74, 128, 36, 136, 22, 21, 71, 25, 6, 23, 124, 79, 38, 132, 93, 18, 112, 86, 30, 135, 39, 114, 55, 63, 13, 102, 59, 9, 17, 16, 65, 64, 138, 56, 57, 99, 97, 72, 50, 125, 45, 70, 66, 53, 123, 127, 94, 61, 43, 51, 90, 92, 29, 119, 108, 27, 60, 46, 40, 107, 20, 47, 96, 1, 76, 111, 131, 87, 120, 32, 4, 137, 139, 37, 106, 95, 103, 121, 14, 129, 12, 69, 2, 105, 5, 48, 118, 28, 33, 133, 88, 134, 19, 116, 130, 68, 73, 34, 101, 7, 24, 11, 83, 126, 58, 100, 117, 42, 109, 115, 78, 10, 98],
        [128, 72, 19, 26, 35, 59, 137, 18, 121, 64, 23, 10, 40, 125, 115, 71, 89, 113, 130, 57, 12, 39, 16, 116, 108, 54, 53, 70, 135, 94, 131, 106, 124, 20, 13, 9, 47, 24, 129, 31, 45, 25, 78, 44, 66, 28, 56, 100, 2, 0, 77, 52, 98, 30, 87, 1, 17, 118, 138, 48, 134, 7, 65, 79, 41, 63, 50, 107, 61, 99, 109, 136, 95, 81, 92, 74, 15, 139, 37, 119, 73, 86, 133, 114, 120, 49, 82, 117, 103, 51, 122, 111, 29, 91, 60, 104, 97, 80, 58, 96, 126, 62, 4, 123, 42, 83, 6, 101, 127, 85, 102, 36, 55, 90, 88, 132, 76, 5, 11, 34, 8, 84, 67, 38, 46, 27, 3, 93, 69, 112, 14, 110, 105, 22, 33, 32, 75, 68, 21, 43],
        [18, 78, 100, 102, 52, 129, 138, 99, 64, 90, 79, 63, 0, 45, 60, 116, 47, 26, 87, 50, 32, 65, 103, 70, 83, 41, 68, 24, 80, 37, 105, 107, 109, 57, 25, 28, 3, 9, 101, 123, 137, 134, 58, 94, 135, 117, 2, 23, 133, 92, 125, 91, 121, 1, 42, 15, 106, 43, 118, 77, 128, 124, 6, 30, 40, 53, 62, 127, 33, 8, 93, 126, 115, 12, 46, 122, 13, 5, 36, 132, 131, 111, 110, 104, 20, 19, 14, 55, 56, 88, 69, 66, 130, 38, 39, 76, 17, 84, 85, 7, 74, 27, 34, 61, 108, 86, 21, 112, 59, 16, 35, 120, 119, 51, 96, 29, 81, 97, 11, 95, 44, 71, 73, 98, 10, 31, 75, 82, 4, 54, 114, 139, 22, 48, 89, 49, 113, 72, 136, 67],
        [117, 84, 49, 57, 6, 107, 78, 62, 14, 32, 87, 9, 47, 19, 16, 5, 24, 92, 37, 70, 7, 31, 88, 39, 120, 100, 33, 22, 3, 105, 10, 91, 42, 34, 101, 71, 27, 127, 1, 90, 122, 103, 86, 11, 97, 135, 63, 43, 8, 69, 106, 72, 73, 109, 66, 35, 48, 83, 12, 53, 133, 25, 4, 118, 58, 130, 113, 112, 2, 80, 46, 111, 60, 59, 52, 56, 75, 93, 89, 41, 121, 85, 28, 114, 21, 134, 77, 126, 94, 61, 104, 98, 29, 116, 129, 131, 123, 139, 95, 64, 102, 44, 23, 50, 20, 26, 18, 36, 55, 82, 17, 51, 40, 54, 65, 45, 13, 108, 99, 124, 68, 79, 15, 38, 96, 0, 125, 67, 115, 128, 110, 138, 119, 74, 30, 136, 137, 132, 76, 81],
        [7, 28, 43, 82, 76, 116, 40, 73, 44, 94, 24, 107, 119, 52, 92, 70, 97, 118, 125, 39, 59, 60, 102, 111, 133, 106, 49, 99, 96, 1, 51, 110, 35, 113, 105, 112, 77, 85, 114, 72, 108, 29, 41, 122, 91, 98, 25, 21, 54, 46, 0, 6, 109, 81, 20, 68, 19, 37, 34, 132, 127, 55, 137, 120, 47, 64, 2, 65, 11, 80, 103, 69, 61, 88, 117, 15, 71, 16, 32, 53, 89, 8, 38, 86, 134, 126, 139, 136, 5, 128, 9, 4, 131, 27, 45, 95, 50, 87, 62, 123, 66, 93, 17, 115, 121, 56, 124, 33, 12, 18, 57, 79, 48, 42, 26, 104, 22, 30, 78, 90, 83, 13, 36, 101, 75, 135, 31, 14, 84, 10, 130, 138, 100, 74, 63, 3, 23, 129, 67, 58],
        [67, 26, 27, 65, 24, 21, 82, 38, 66, 63, 33, 57, 109, 128, 6, 69, 95, 8, 87, 120, 134, 99, 28, 44, 122, 88, 22, 5, 112, 11, 34, 20, 132, 70, 135, 127, 49, 58, 105, 30, 48, 130, 7, 53, 114, 56, 37, 107, 80, 62, 43, 45, 106, 139, 138, 104, 110, 121, 18, 129, 39, 113, 19, 9, 1, 126, 71, 75, 47, 73, 116, 68, 74, 17, 60, 31, 59, 93, 84, 42, 102, 94, 54, 78, 41, 90, 79, 23, 118, 119, 98, 111, 29, 91, 2, 103, 46, 36, 40, 72, 13, 16, 32, 92, 15, 125, 4, 50, 85, 101, 10, 77, 133, 124, 117, 97, 136, 61, 12, 86, 55, 51, 14, 89, 131, 25, 76, 0, 52, 83, 115, 100, 123, 108, 81, 64, 3, 35, 96, 137],
        [132, 75, 67, 95, 32, 18, 124, 40, 94, 26, 76, 93, 56, 82, 29, 64, 112, 78, 0, 27, 92, 49, 50, 71, 91, 65, 111, 44, 97, 46, 105, 58, 2, 51, 53, 14, 113, 117, 96, 100, 4, 11, 90, 6, 119, 118, 13, 31, 122, 19, 79, 37, 98, 137, 24, 72, 45, 39, 9, 59, 109, 130, 66, 106, 115, 61, 77, 110, 73, 10, 12, 129, 8, 108, 138, 57, 70, 36, 42, 34, 69, 102, 139, 60, 28, 135, 127, 131, 87, 83, 47, 125, 16, 80, 68, 35, 114, 126, 41, 20, 1, 133, 3, 23, 43, 81, 33, 89, 17, 52, 54, 5, 85, 88, 21, 104, 7, 15, 101, 121, 30, 128, 134, 116, 63, 25, 86, 99, 74, 55, 103, 120, 136, 123, 107, 48, 62, 22, 84, 38],
        [39, 9, 74, 16, 83, 138, 26, 65, 103, 84, 93, 130, 76, 68, 102, 119, 1, 44, 10, 123, 85, 11, 29, 0, 116, 113, 37, 120, 43, 67, 137, 109, 31, 36, 66, 73, 110, 125, 54, 63, 14, 96, 135, 122, 13, 34, 18, 22, 101, 87, 4, 90, 38, 104, 15, 53, 41, 118, 21, 139, 99, 62, 134, 42, 124, 95, 94, 89, 100, 91, 82, 2, 81, 48, 133, 112, 115, 129, 52, 6, 28, 24, 57, 56, 33, 59, 107, 55, 117, 49, 70, 51, 98, 127, 71, 108, 72, 27, 32, 20, 131, 17, 19, 61, 92, 64, 77, 58, 132, 80, 97, 75, 46, 79, 12, 126, 3, 69, 88, 60, 5, 128, 136, 86, 47, 105, 106, 50, 78, 40, 111, 23, 45, 35, 25, 121, 8, 30, 7, 114],
        [136, 65, 126, 119, 96, 59, 8, 35, 132, 30, 124, 20, 113, 36, 100, 29, 6, 78, 86, 67, 1, 17, 129, 107, 15, 84, 106, 5, 4, 114, 58, 120, 111, 12, 77, 127, 66, 23, 90, 26, 91, 63, 53, 46, 18, 10, 102, 9, 56, 32, 97, 116, 117, 133, 70, 27, 105, 81, 14, 76, 123, 135, 34, 110, 68, 98, 109, 11, 52, 57, 3, 88, 85, 103, 69, 43, 31, 139, 118, 92, 40, 137, 45, 61, 7, 73, 72, 51, 13, 54, 95, 50, 128, 79, 48, 75, 87, 93, 115, 38, 121, 64, 108, 42, 99, 19, 33, 47, 41, 24, 94, 138, 83, 104, 71, 21, 22, 28, 37, 131, 130, 134, 2, 39, 44, 25, 112, 0, 125, 122, 74, 80, 49, 60, 82, 16, 62, 89, 55, 101],
        [122, 96, 20, 28, 92, 98, 121, 52, 54, 91, 55, 127, 88, 109, 111, 6, 25, 14, 51, 49, 63, 24, 29, 15, 132, 136, 101, 84, 23, 21, 139, 77, 53, 27, 42, 30, 40, 83, 80, 117, 75, 1, 22, 133, 37, 86, 76, 130, 74, 41, 60, 95, 35, 103, 17, 71, 19, 56, 59, 89, 138, 5, 82, 134, 43, 78, 70, 72, 73, 128, 61, 129, 87, 125, 45, 38, 93, 85, 36, 123, 9, 66, 116, 124, 16, 58, 135, 65, 48, 44, 33, 108, 3, 90, 64, 50, 10, 18, 94, 107, 97, 79, 105, 137, 113, 100, 112, 47, 131, 62, 119, 67, 2, 99, 11, 118, 39, 69, 57, 115, 31, 106, 26, 32, 104, 4, 68, 102, 81, 46, 0, 34, 110, 12, 114, 126, 8, 13, 120, 7],
        [72, 80, 66, 78, 112, 7, 43, 86, 113, 135, 41, 87, 27, 42, 18, 96, 59, 121, 17, 116, 56, 90, 39, 55, 98, 65, 91, 31, 29, 68, 34, 63, 47, 103, 25, 33, 26, 8, 137, 30, 46, 61, 123, 82, 58, 139, 108, 24, 16, 122, 10, 70, 44, 101, 20, 51, 106, 36, 21, 69, 129, 84, 4, 105, 131, 126, 75, 104, 120, 0, 117, 35, 138, 119, 76, 125, 32, 71, 11, 14, 12, 3, 57, 67, 128, 6, 45, 9, 109, 97, 94, 49, 13, 133, 37, 81, 118, 85, 52, 83, 38, 134, 60, 110, 50, 23, 40, 132, 64, 127, 1, 22, 114, 115, 107, 124, 48, 19, 136, 15, 74, 28, 53, 93, 100, 89, 88, 5, 99, 92, 102, 73, 79, 54, 77, 62, 95, 111, 130, 2],
        [73, 82, 52, 103, 60, 83, 135, 45, 85, 2, 48, 31, 114, 95, 44, 20, 120, 88, 107, 18, 46, 1, 117, 139, 70, 64, 54, 57, 118, 134, 15, 41, 79, 71, 14, 137, 28, 84, 97, 26, 106, 4, 5, 121, 96, 47, 127, 58, 38, 17, 86, 133, 136, 35, 122, 92, 124, 63, 108, 11, 104, 68, 65, 56, 50, 77, 7, 0, 80, 39, 3, 123, 115, 112, 110, 94, 78, 19, 72, 87, 129, 116, 8, 90, 109, 59, 16, 6, 81, 62, 21, 74, 24, 111, 36, 23, 22, 30, 34, 25, 61, 89, 113, 32, 130, 126, 98, 131, 128, 132, 69, 99, 75, 101, 29, 105, 125, 40, 119, 66, 49, 51, 55, 10, 102, 76, 12, 37, 33, 91, 9, 13, 27, 43, 138, 100, 93, 42, 53, 67],
        [87, 29, 53, 10, 0, 14, 96, 8, 109, 123, 76, 72, 114, 5, 137, 39, 9, 93, 118, 75, 85, 77, 91, 104, 115, 74, 11, 3, 121, 2, 113, 98, 139, 58, 84, 81, 59, 102, 82, 120, 60, 106, 27, 89, 105, 67, 41, 32, 33, 116, 36, 35, 92, 23, 68, 37, 51, 30, 71, 88, 21, 130, 126, 42, 65, 25, 95, 56, 110, 47, 49, 97, 34, 78, 131, 63, 40, 133, 90, 69, 61, 122, 136, 125, 138, 54, 64, 26, 16, 17, 19, 80, 94, 100, 22, 62, 28, 86, 13, 46, 43, 99, 132, 128, 73, 134, 101, 119, 44, 45, 1, 112, 70, 127, 79, 50, 108, 135, 111, 12, 129, 38, 103, 117, 20, 15, 48, 107, 18, 124, 7, 24, 52, 66, 6, 83, 31, 4, 57, 55],
        [72, 92, 77, 136, 79, 14, 40, 0, 58, 67, 93, 3, 61, 66, 19, 55, 97, 50, 51, 76, 134, 41, 102, 85, 42, 101, 20, 123, 115, 94, 138, 63, 118, 26, 81, 39, 88, 116, 15, 49, 62, 25, 90, 28, 44, 54, 52, 128, 6, 68, 47, 137, 113, 46, 22, 27, 1, 13, 9, 71, 70, 98, 89, 124, 135, 104, 100, 59, 11, 133, 73, 43, 29, 12, 65, 91, 129, 16, 32, 78, 31, 112, 18, 56, 125, 48, 103, 87, 114, 110, 109, 126, 117, 99, 24, 86, 21, 127, 139, 69, 10, 5, 131, 4, 23, 82, 2, 36, 35, 119, 83, 111, 53, 121, 80, 120, 7, 45, 75, 57, 107, 130, 84, 33, 132, 108, 8, 37, 38, 17, 96, 34, 106, 60, 105, 74, 30, 95, 64, 122],
        [65, 23, 119, 106, 17, 96, 104, 62, 129, 90, 48, 66, 68, 112, 21, 74, 6, 117, 4, 14, 52, 139, 101, 135, 25, 87, 137, 54, 95, 1, 108, 20, 134, 27, 11, 121, 89, 100, 15, 102, 133, 98, 49, 105, 32, 69, 79, 16, 91, 94, 72, 123, 77, 81, 22, 138, 130, 44, 29, 67, 2, 92, 13, 37, 132, 127, 45, 63, 46, 36, 115, 82, 93, 85, 114, 131, 83, 42, 109, 24, 57, 43, 80, 26, 73, 124, 39, 9, 113, 58, 78, 12, 70, 103, 5, 53, 18, 41, 10, 64, 47, 59, 110, 34, 125, 61, 40, 116, 111, 7, 76, 33, 71, 51, 86, 122, 84, 35, 55, 120, 128, 3, 8, 56, 126, 38, 88, 28, 60, 50, 99, 75, 31, 97, 136, 30, 118, 0, 19, 107],
        [64, 114, 72, 8, 31, 81, 75, 46, 132, 138, 57, 91, 125, 28, 15, 116, 48, 33, 40, 30, 135, 76, 128, 20, 89, 74, 25, 109, 121, 112, 88, 85, 24, 49, 94, 99, 133, 7, 68, 1, 129, 13, 39, 51, 134, 17, 44, 66, 104, 105, 96, 98, 23, 137, 35, 50, 92, 101, 6, 77, 29, 2, 122, 136, 103, 100, 52, 65, 63, 70, 37, 21, 14, 124, 90, 110, 27, 41, 9, 79, 16, 4, 108, 38, 93, 95, 118, 62, 83, 119, 78, 139, 34, 5, 61, 54, 53, 59, 56, 117, 60, 73, 22, 120, 47, 111, 87, 32, 43, 84, 127, 45, 55, 113, 106, 102, 10, 69, 97, 123, 71, 80, 67, 131, 3, 0, 82, 130, 18, 36, 42, 58, 26, 115, 126, 12, 11, 86, 19, 107],
        [81, 84, 54, 83, 75, 58, 52, 27, 111, 25, 78, 71, 136, 67, 20, 100, 97, 62, 22, 138, 127, 134, 109, 13, 66, 14, 33, 42, 35, 2, 95, 92, 101, 17, 49, 39, 44, 120, 28, 112, 30, 16, 21, 37, 19, 110, 4, 23, 116, 57, 86, 118, 103, 31, 74, 132, 123, 125, 32, 7, 99, 6, 124, 38, 46, 64, 77, 29, 119, 131, 50, 80, 117, 108, 5, 68, 107, 26, 1, 96, 61, 129, 122, 72, 69, 3, 135, 137, 51, 98, 79, 18, 121, 56, 0, 113, 139, 36, 48, 82, 9, 106, 53, 73, 126, 90, 59, 76, 60, 105, 55, 133, 85, 114, 102, 12, 130, 115, 104, 128, 47, 91, 10, 93, 41, 24, 11, 15, 87, 8, 65, 40, 45, 63, 34, 88, 89, 94, 43, 70],
        [131, 118, 26, 72, 14, 27, 132, 55, 117, 101, 0, 66, 114, 65, 13, 62, 84, 119, 34, 99, 42, 18, 35, 135, 79, 123, 88, 97, 93, 43, 77, 100, 8, 44, 33, 46, 107, 64, 98, 102, 136, 15, 17, 71, 133, 49, 112, 54, 20, 32, 137, 11, 130, 86, 31, 7, 87, 3, 4, 39, 51, 125, 89, 70, 108, 75, 9, 36, 113, 24, 115, 127, 16, 83, 90, 61, 74, 37, 69, 106, 29, 50, 59, 30, 111, 2, 38, 110, 128, 12, 103, 56, 85, 104, 81, 40, 1, 63, 67, 76, 68, 129, 6, 138, 124, 105, 82, 10, 126, 58, 134, 45, 73, 19, 21, 57, 92, 53, 23, 96, 95, 47, 94, 121, 78, 5, 91, 22, 109, 48, 122, 60, 25, 80, 139, 116, 120, 28, 52, 41],
        [64, 17, 32, 31, 138, 102, 30, 71, 5, 19, 109, 49, 40, 47, 34, 41, 14, 50, 104, 135, 27, 76, 112, 15, 10, 126, 1, 129, 107, 114, 18, 42, 110, 69, 136, 123, 72, 130, 77, 39, 132, 99, 67, 20, 33, 98, 46, 108, 127, 70, 137, 6, 84, 94, 12, 2, 116, 7, 0, 119, 61, 45, 90, 59, 106, 3, 57, 121, 16, 37, 13, 62, 78, 74, 28, 96, 66, 133, 93, 124, 95, 100, 35, 22, 54, 58, 111, 88, 81, 43, 87, 86, 101, 92, 11, 105, 131, 65, 117, 8, 23, 51, 113, 60, 82, 38, 73, 48, 9, 91, 122, 103, 120, 55, 89, 85, 24, 97, 83, 80, 29, 44, 134, 118, 139, 52, 53, 36, 79, 56, 125, 115, 68, 4, 25, 21, 128, 63, 75, 26],
        [72, 117, 69, 112, 136, 46, 123, 61, 79, 82, 91, 121, 111, 109, 5, 114, 89, 41, 20, 6, 22, 93, 131, 74, 86, 128, 100, 53, 77, 35, 24, 45, 68, 73, 129, 119, 26, 31, 85, 7, 51, 65, 50, 71, 30, 84, 18, 130, 78, 96, 2, 80, 63, 67, 107, 83, 49, 39, 59, 1, 12, 28, 9, 38, 58, 4, 105, 75, 42, 87, 0, 127, 11, 54, 101, 104, 133, 52, 32, 17, 94, 15, 95, 137, 64, 115, 55, 34, 118, 106, 27, 43, 90, 99, 3, 113, 47, 116, 29, 88, 23, 62, 57, 13, 56, 10, 122, 97, 120, 103, 14, 25, 21, 40, 125, 139, 98, 126, 37, 92, 48, 134, 70, 135, 124, 16, 132, 36, 60, 19, 44, 81, 76, 33, 138, 66, 8, 102, 108, 110],
        [43, 62, 127, 97, 52, 61, 109, 72, 84, 48, 35, 74, 104, 63, 41, 26, 23, 122, 40, 124, 82, 56, 133, 33, 36, 12, 90, 130, 32, 77, 99, 76, 28, 13, 118, 108, 25, 81, 69, 22, 93, 37, 55, 71, 117, 45, 79, 30, 20, 15, 44, 3, 80, 101, 85, 17, 21, 67, 39, 135, 66, 8, 0, 73, 106, 103, 57, 6, 1, 131, 60, 107, 19, 123, 38, 98, 58, 113, 105, 102, 119, 139, 120, 42, 18, 100, 126, 121, 31, 137, 53, 5, 51, 27, 94, 95, 136, 128, 96, 75, 112, 50, 54, 129, 91, 2, 78, 24, 68, 11, 10, 46, 65, 29, 88, 9, 14, 47, 87, 116, 4, 59, 64, 110, 86, 49, 134, 114, 115, 125, 70, 16, 92, 83, 132, 138, 111, 34, 7, 89],
        [99, 80, 126, 84, 115, 23, 65, 91, 31, 78, 120, 28, 97, 77, 136, 19, 98, 32, 71, 6, 27, 138, 96, 102, 113, 107, 82, 81, 90, 39, 75, 21, 52, 50, 56, 2, 109, 76, 93, 14, 119, 35, 110, 15, 124, 137, 64, 12, 121, 66, 40, 0, 123, 104, 103, 70, 134, 105, 41, 89, 24, 5, 47, 133, 94, 74, 100, 9, 46, 128, 69, 51, 33, 83, 73, 112, 87, 130, 88, 85, 34, 44, 42, 86, 62, 18, 48, 53, 49, 139, 118, 36, 54, 55, 116, 63, 131, 8, 60, 4, 43, 61, 10, 117, 3, 129, 17, 114, 57, 22, 122, 26, 30, 72, 95, 29, 25, 106, 38, 125, 67, 135, 16, 20, 45, 108, 7, 101, 58, 1, 37, 13, 111, 68, 79, 11, 132, 92, 59, 127],
        [24, 77, 84, 106, 50, 15, 11, 37, 38, 103, 82, 70, 130, 78, 17, 51, 19, 129, 26, 31, 97, 95, 65, 8, 58, 7, 123, 47, 55, 42, 124, 137, 120, 80, 73, 105, 60, 128, 67, 92, 43, 30, 71, 62, 14, 117, 93, 109, 135, 83, 21, 131, 52, 132, 20, 121, 86, 113, 98, 122, 3, 45, 134, 89, 99, 22, 133, 39, 1, 75, 102, 110, 139, 40, 49, 116, 46, 104, 28, 5, 16, 4, 68, 125, 114, 127, 96, 25, 79, 119, 87, 29, 57, 69, 41, 63, 107, 61, 112, 48, 138, 18, 10, 0, 118, 136, 36, 101, 6, 23, 115, 94, 33, 32, 12, 34, 27, 111, 64, 2, 35, 90, 108, 126, 56, 100, 88, 72, 54, 44, 91, 81, 66, 85, 13, 59, 53, 76, 9, 74],
        [98, 129, 128, 89, 95, 1, 81, 0, 66, 47, 36, 139, 133, 16, 44, 78, 110, 45, 35, 92, 115, 4, 26, 77, 72, 75, 116, 8, 37, 11, 42, 88, 3, 94, 32, 65, 125, 57, 126, 85, 103, 109, 41, 64, 29, 69, 122, 9, 10, 117, 38, 91, 123, 121, 76, 12, 132, 82, 63, 84, 80, 99, 39, 21, 51, 134, 17, 138, 102, 97, 2, 56, 104, 61, 55, 60, 7, 124, 100, 19, 22, 111, 137, 27, 14, 106, 34, 70, 118, 83, 68, 71, 53, 67, 31, 136, 127, 93, 48, 25, 114, 107, 62, 86, 49, 54, 131, 79, 87, 33, 40, 113, 50, 73, 112, 5, 13, 20, 135, 23, 101, 130, 43, 6, 24, 46, 28, 74, 15, 18, 58, 96, 119, 120, 59, 90, 105, 52, 30, 108],
        [42, 62, 72, 21, 55, 25, 112, 22, 31, 125, 136, 41, 102, 9, 81, 24, 23, 101, 33, 88, 94, 128, 6, 28, 93, 30, 52, 111, 119, 122, 73, 97, 43, 131, 104, 2, 82, 133, 65, 75, 18, 105, 70, 137, 124, 80, 127, 107, 48, 132, 14, 60, 134, 38, 85, 118, 11, 45, 47, 139, 96, 54, 99, 100, 53, 5, 114, 79, 113, 16, 71, 117, 59, 34, 126, 8, 68, 56, 51, 90, 110, 66, 138, 35, 39, 44, 92, 37, 50, 86, 76, 95, 121, 116, 1, 120, 0, 36, 17, 108, 15, 89, 10, 87, 57, 135, 61, 84, 46, 109, 29, 103, 78, 123, 40, 130, 69, 115, 129, 32, 106, 83, 12, 20, 4, 91, 3, 49, 7, 27, 13, 19, 63, 74, 98, 58, 26, 64, 77, 67],
        [107, 30, 35, 104, 135, 66, 133, 87, 68, 89, 64, 91, 50, 94, 13, 0, 124, 112, 70, 25, 65, 75, 116, 23, 56, 38, 71, 29, 31, 80, 83, 118, 69, 81, 117, 15, 26, 10, 78, 4, 101, 115, 100, 21, 95, 109, 129, 44, 114, 58, 36, 128, 126, 37, 16, 6, 76, 136, 122, 84, 131, 9, 19, 17, 42, 63, 34, 74, 110, 123, 40, 96, 28, 138, 72, 2, 97, 127, 39, 53, 11, 46, 73, 85, 134, 33, 119, 51, 99, 1, 55, 125, 90, 8, 54, 32, 5, 120, 27, 52, 43, 61, 82, 132, 3, 59, 57, 60, 137, 47, 88, 86, 48, 77, 79, 108, 14, 93, 98, 130, 62, 92, 7, 41, 103, 24, 139, 102, 105, 18, 121, 20, 113, 111, 12, 22, 106, 49, 45, 67],
        [133, 125, 49, 59, 32, 128, 130, 99, 56, 69, 70, 37, 104, 39, 19, 126, 8, 123, 105, 124, 47, 27, 139, 89, 0, 110, 54, 50, 6, 71, 131, 55, 138, 63, 91, 102, 95, 96, 42, 44, 48, 88, 113, 17, 1, 31, 9, 3, 82, 119, 13, 137, 58, 4, 34, 16, 127, 46, 106, 64, 24, 30, 23, 67, 66, 85, 72, 53, 118, 135, 18, 116, 14, 40, 51, 100, 36, 111, 90, 78, 112, 114, 80, 79, 21, 132, 101, 81, 28, 83, 10, 29, 107, 26, 86, 108, 115, 22, 35, 2, 77, 84, 68, 109, 75, 11, 57, 38, 12, 120, 45, 73, 5, 98, 129, 136, 134, 76, 61, 103, 92, 122, 60, 97, 41, 25, 94, 93, 15, 33, 43, 117, 87, 7, 62, 121, 74, 52, 65, 20],
        [56, 118, 25, 23, 86, 74, 63, 12, 41, 68, 131, 97, 59, 69, 15, 73, 34, 81, 106, 115, 44, 42, 8, 129, 110, 5, 39, 77, 127, 92, 105, 38, 26, 13, 125, 119, 48, 107, 126, 54, 9, 75, 19, 100, 31, 121, 47, 33, 98, 113, 64, 67, 90, 49, 83, 120, 27, 103, 87, 70, 89, 133, 82, 123, 94, 43, 65, 21, 3, 85, 135, 40, 46, 137, 124, 18, 101, 22, 91, 60, 57, 93, 51, 58, 84, 1, 0, 122, 4, 114, 2, 138, 11, 111, 17, 130, 55, 139, 79, 88, 132, 72, 99, 50, 37, 116, 80, 78, 30, 96, 10, 36, 35, 104, 117, 112, 20, 76, 109, 62, 95, 16, 134, 45, 24, 7, 128, 32, 52, 53, 66, 136, 6, 61, 29, 14, 71, 28, 108, 102],
        [66, 49, 59, 36, 19, 123, 103, 5, 17, 3, 133, 84, 109, 67, 119, 60, 50, 89, 51, 33, 134, 62, 8, 40, 12, 111, 15, 57, 96, 100, 6, 88, 116, 132, 37, 101, 138, 87, 139, 46, 131, 28, 97, 94, 115, 79, 0, 13, 90, 78, 108, 85, 7, 129, 122, 32, 126, 54, 47, 44, 1, 75, 26, 114, 64, 93, 18, 83, 29, 107, 80, 112, 63, 70, 34, 128, 53, 58, 25, 95, 68, 125, 69, 106, 117, 113, 124, 74, 22, 45, 41, 24, 135, 137, 56, 31, 11, 9, 76, 65, 39, 30, 38, 48, 21, 23, 110, 91, 98, 14, 120, 10, 99, 86, 130, 16, 2, 118, 92, 77, 27, 121, 61, 81, 43, 20, 35, 71, 105, 82, 104, 72, 55, 52, 127, 4, 136, 73, 102, 42],
        [77, 100, 81, 127, 138, 93, 37, 13, 55, 23, 68, 114, 137, 139, 24, 133, 14, 129, 67, 135, 123, 101, 31, 82, 91, 48, 66, 20, 111, 119, 124, 3, 52, 29, 65, 136, 43, 83, 130, 76, 115, 96, 17, 110, 74, 32, 25, 44, 80, 107, 8, 57, 79, 131, 49, 99, 39, 41, 122, 18, 120, 112, 19, 4, 7, 109, 90, 125, 45, 98, 72, 106, 11, 73, 97, 54, 26, 46, 2, 69, 103, 9, 28, 47, 53, 10, 15, 62, 84, 94, 5, 40, 51, 34, 71, 61, 108, 42, 1, 88, 21, 86, 6, 33, 0, 95, 116, 50, 126, 38, 63, 78, 117, 56, 102, 36, 87, 105, 12, 121, 134, 35, 70, 118, 128, 59, 92, 75, 113, 89, 58, 85, 16, 132, 104, 27, 30, 64, 22, 60],
        [50, 39, 102, 101, 7, 41, 66, 44, 25, 139, 6, 63, 121, 126, 78, 133, 1, 116, 13, 12, 74, 38, 35, 46, 93, 30, 42, 123, 130, 81, 5, 124, 119, 105, 113, 114, 136, 125, 77, 107, 91, 11, 17, 60, 138, 55, 19, 18, 100, 71, 16, 104, 86, 27, 115, 95, 76, 131, 20, 31, 4, 65, 94, 129, 106, 99, 103, 110, 128, 59, 87, 0, 9, 54, 90, 51, 70, 57, 8, 92, 62, 84, 24, 23, 96, 2, 61, 112, 36, 89, 49, 82, 3, 67, 29, 72, 26, 69, 64, 15, 58, 43, 98, 137, 97, 56, 134, 73, 111, 52, 21, 22, 32, 127, 79, 68, 83, 132, 108, 40, 75, 120, 48, 45, 33, 135, 53, 34, 122, 80, 14, 118, 88, 109, 47, 10, 28, 85, 37, 117],
        [23, 97, 5, 92, 122, 38, 107, 13, 18, 126, 64, 37, 88, 45, 3, 58, 130, 124, 102, 21, 32, 79, 57, 42, 76, 62, 94, 29, 56, 95, 98, 40, 106, 85, 104, 83, 118, 2, 134, 110, 96, 46, 31, 66, 99, 0, 61, 16, 43, 20, 112, 132, 71, 68, 72, 93, 100, 39, 50, 10, 52, 101, 133, 67, 82, 70, 86, 19, 53, 137, 28, 7, 41, 90, 91, 114, 27, 36, 131, 135, 47, 128, 1, 51, 9, 22, 105, 6, 109, 63, 108, 35, 127, 12, 75, 44, 136, 55, 125, 14, 87, 69, 11, 77, 30, 54, 138, 120, 4, 117, 33, 17, 80, 115, 60, 24, 8, 26, 65, 123, 89, 111, 49, 121, 34, 119, 113, 81, 25, 139, 129, 84, 59, 103, 78, 116, 74, 15, 73, 48],
        [43, 64, 95, 110, 14, 17, 29, 41, 57, 26, 79, 53, 87, 16, 102, 113, 60, 52, 119, 116, 50, 30, 82, 81, 132, 51, 65, 67, 123, 47, 23, 107, 92, 99, 84, 13, 128, 37, 126, 127, 15, 24, 88, 34, 73, 49, 118, 74, 35, 114, 124, 98, 109, 72, 42, 83, 89, 10, 134, 8, 22, 2, 105, 130, 115, 9, 129, 20, 1, 32, 21, 117, 3, 104, 31, 106, 45, 56, 100, 66, 96, 112, 18, 28, 33, 25, 136, 103, 122, 137, 85, 48, 38, 71, 86, 44, 59, 90, 125, 40, 120, 101, 54, 138, 5, 7, 63, 76, 58, 135, 77, 0, 62, 55, 133, 36, 27, 80, 108, 19, 61, 94, 46, 12, 78, 4, 131, 69, 121, 70, 6, 68, 75, 93, 91, 39, 97, 139, 11, 111],
        [14, 58, 103, 25, 42, 115, 77, 44, 130, 127, 22, 62, 97, 84, 135, 45, 87, 13, 53, 137, 116, 122, 54, 74, 126, 15, 138, 89, 32, 2, 59, 80, 27, 36, 118, 66, 100, 134, 111, 19, 90, 51, 75, 39, 29, 82, 124, 131, 3, 33, 128, 63, 47, 85, 91, 17, 52, 56, 28, 38, 49, 40, 64, 121, 31, 110, 10, 107, 50, 81, 72, 113, 61, 43, 73, 55, 114, 11, 57, 23, 88, 83, 96, 98, 136, 34, 37, 92, 117, 1, 102, 24, 65, 8, 101, 16, 0, 46, 48, 67, 30, 12, 70, 71, 94, 93, 35, 7, 104, 21, 18, 9, 132, 95, 112, 5, 109, 76, 123, 4, 6, 129, 60, 99, 78, 68, 120, 69, 41, 26, 119, 79, 106, 86, 105, 133, 125, 108, 139, 20],
        [51, 85, 95, 26, 64, 92, 98, 70, 15, 66, 106, 37, 94, 62, 133, 67, 20, 119, 108, 9, 120, 74, 111, 14, 76, 104, 129, 82, 61, 127, 60, 118, 58, 105, 77, 102, 17, 0, 40, 56, 130, 72, 46, 1, 110, 45, 139, 48, 18, 34, 5, 24, 41, 124, 49, 47, 38, 126, 114, 93, 33, 7, 112, 21, 79, 132, 30, 107, 2, 42, 39, 81, 19, 6, 57, 12, 131, 125, 121, 123, 101, 3, 69, 68, 91, 22, 100, 138, 78, 59, 54, 35, 86, 80, 4, 134, 109, 25, 103, 96, 116, 27, 53, 16, 115, 65, 113, 136, 128, 13, 75, 88, 137, 90, 135, 31, 11, 73, 83, 43, 117, 32, 8, 10, 23, 29, 97, 63, 50, 84, 99, 122, 87, 28, 44, 71, 36, 55, 52, 89],
        [75, 28, 114, 129, 78, 4, 55, 5, 89, 20, 39, 40, 106, 96, 120, 7, 36, 65, 41, 137, 76, 136, 1, 29, 13, 52, 10, 93, 88, 8, 74, 122, 0, 123, 30, 73, 115, 103, 108, 113, 58, 60, 116, 16, 51, 72, 56, 63, 99, 22, 2, 59, 9, 35, 81, 117, 135, 67, 38, 107, 53, 70, 80, 104, 34, 27, 134, 15, 94, 26, 85, 111, 24, 66, 95, 119, 18, 21, 97, 139, 12, 100, 125, 90, 133, 23, 61, 124, 47, 128, 112, 57, 105, 62, 126, 25, 19, 118, 87, 83, 69, 37, 138, 131, 17, 3, 42, 43, 71, 130, 86, 82, 77, 31, 92, 132, 33, 101, 102, 32, 98, 46, 48, 54, 91, 68, 49, 11, 121, 14, 50, 44, 127, 110, 84, 79, 6, 109, 45, 64],
        [78, 46, 10, 99, 51, 69, 124, 55, 114, 88, 0, 32, 4, 2, 61, 35, 63, 106, 136, 73, 89, 48, 84, 81, 45, 50, 137, 79, 36, 121, 120, 31, 133, 14, 102, 28, 92, 38, 74, 95, 49, 22, 127, 20, 97, 77, 41, 6, 91, 15, 139, 9, 27, 90, 53, 59, 138, 37, 5, 119, 80, 82, 16, 132, 112, 87, 86, 52, 107, 135, 129, 76, 12, 44, 19, 34, 1, 40, 25, 130, 85, 3, 56, 23, 111, 54, 43, 68, 93, 11, 98, 72, 39, 94, 8, 113, 125, 104, 110, 96, 26, 70, 100, 7, 103, 123, 122, 42, 21, 30, 33, 131, 17, 126, 58, 128, 13, 65, 62, 118, 75, 101, 71, 115, 66, 24, 116, 108, 29, 105, 57, 134, 47, 117, 109, 64, 18, 60, 83, 67],
        [118, 132, 79, 126, 113, 120, 36, 26, 109, 67, 41, 13, 77, 91, 35, 127, 80, 139, 20, 21, 81, 75, 135, 47, 123, 27, 122, 44, 56, 129, 134, 101, 85, 54, 48, 62, 125, 99, 15, 138, 100, 25, 10, 107, 115, 2, 116, 64, 18, 70, 29, 30, 5, 131, 108, 46, 22, 49, 7, 59, 52, 76, 34, 58, 103, 40, 65, 133, 121, 105, 102, 73, 31, 66, 61, 1, 33, 19, 90, 50, 28, 93, 71, 137, 9, 96, 88, 3, 98, 38, 84, 0, 69, 42, 55, 97, 11, 39, 53, 4, 14, 43, 45, 106, 136, 110, 94, 92, 12, 17, 37, 57, 23, 117, 112, 111, 87, 8, 82, 114, 89, 24, 130, 83, 32, 51, 128, 124, 63, 68, 78, 74, 60, 104, 72, 16, 86, 6, 95, 119],
        [108, 126, 49, 80, 127, 2, 44, 73, 113, 35, 78, 88, 74, 135, 19, 21, 38, 84, 125, 111, 124, 71, 100, 139, 25, 95, 33, 133, 81, 104, 116, 48, 39, 0, 112, 97, 61, 23, 63, 136, 62, 119, 51, 31, 85, 68, 67, 130, 134, 45, 5, 46, 132, 42, 77, 57, 122, 101, 91, 27, 8, 138, 16, 105, 17, 22, 115, 118, 109, 89, 52, 83, 87, 4, 55, 75, 72, 3, 131, 128, 6, 34, 94, 70, 40, 12, 14, 117, 96, 90, 110, 102, 43, 66, 9, 30, 103, 20, 13, 82, 37, 86, 47, 120, 32, 79, 28, 7, 107, 58, 56, 121, 24, 36, 41, 26, 76, 10, 114, 11, 1, 65, 93, 137, 99, 59, 15, 18, 69, 29, 92, 60, 123, 54, 64, 53, 98, 106, 129, 50],
        [135, 79, 82, 61, 103, 43, 112, 38, 46, 127, 67, 134, 64, 6, 40, 7, 30, 23, 78, 52, 44, 104, 97, 122, 86, 53, 71, 37, 49, 57, 130, 33, 121, 119, 36, 41, 16, 19, 58, 88, 132, 10, 115, 129, 62, 59, 102, 25, 32, 69, 85, 24, 93, 117, 70, 94, 124, 56, 55, 48, 74, 21, 110, 66, 92, 128, 84, 133, 75, 95, 8, 26, 87, 118, 111, 99, 14, 51, 63, 68, 72, 9, 15, 98, 35, 89, 17, 105, 54, 96, 22, 126, 73, 20, 109, 5, 100, 4, 138, 60, 83, 42, 116, 34, 80, 47, 18, 50, 31, 136, 11, 90, 0, 120, 131, 101, 28, 81, 113, 107, 2, 139, 39, 91, 76, 1, 29, 125, 12, 65, 3, 13, 45, 123, 27, 114, 77, 108, 137, 106],
        [45, 113, 135, 111, 8, 137, 20, 47, 59, 2, 86, 40, 48, 102, 74, 131, 3, 133, 50, 39, 37, 83, 79, 63, 124, 97, 89, 68, 33, 7, 13, 100, 16, 36, 75, 26, 17, 112, 52, 35, 56, 4, 127, 120, 99, 30, 6, 126, 34, 61, 65, 95, 136, 106, 73, 19, 116, 77, 25, 42, 122, 123, 94, 80, 82, 130, 31, 41, 108, 104, 125, 96, 139, 51, 118, 10, 90, 38, 15, 98, 5, 23, 22, 29, 110, 91, 134, 85, 84, 27, 70, 115, 32, 28, 43, 67, 21, 119, 66, 53, 24, 72, 78, 58, 93, 121, 92, 88, 62, 9, 14, 132, 55, 11, 76, 129, 107, 114, 71, 60, 18, 0, 46, 117, 54, 12, 44, 81, 57, 103, 101, 109, 49, 1, 69, 138, 105, 87, 64, 128],
        [25, 59, 12, 48, 112, 10, 105, 3, 113, 77, 69, 108, 104, 65, 87, 84, 22, 50, 99, 133, 4, 117, 57, 66, 100, 130, 98, 139, 35, 27, 1, 123, 122, 29, 72, 30, 11, 52, 5, 56, 109, 70, 97, 21, 68, 51, 15, 13, 82, 18, 74, 62, 31, 37, 7, 116, 55, 2, 129, 125, 23, 54, 101, 58, 9, 64, 19, 102, 138, 32, 17, 136, 96, 53, 40, 110, 39, 41, 75, 14, 20, 128, 119, 135, 94, 81, 26, 73, 42, 60, 118, 92, 71, 89, 79, 38, 61, 121, 80, 24, 85, 126, 134, 106, 132, 90, 83, 63, 78, 131, 86, 49, 93, 111, 8, 47, 16, 76, 95, 6, 107, 28, 34, 114, 127, 120, 103, 43, 67, 33, 137, 44, 45, 0, 46, 115, 124, 91, 36, 88],
        [64, 58, 36, 44, 40, 68, 123, 20, 139, 122, 65, 79, 16, 111, 8, 17, 125, 61, 130, 2, 56, 120, 72, 63, 134, 1, 12, 21, 127, 82, 114, 33, 96, 27, 137, 50, 99, 117, 14, 76, 48, 49, 102, 9, 25, 53, 70, 18, 73, 31, 4, 46, 129, 83, 13, 10, 128, 22, 3, 35, 66, 95, 41, 110, 133, 55, 108, 15, 136, 118, 80, 5, 81, 26, 106, 138, 11, 6, 51, 91, 86, 113, 71, 100, 121, 0, 104, 19, 84, 54, 90, 85, 88, 45, 93, 67, 112, 75, 119, 92, 30, 34, 38, 60, 77, 116, 69, 32, 126, 52, 62, 98, 57, 109, 107, 23, 39, 43, 74, 42, 59, 105, 29, 115, 97, 47, 87, 101, 24, 135, 7, 28, 94, 132, 124, 131, 37, 103, 78, 89],
        [57, 124, 95, 135, 96, 5, 132, 122, 3, 55, 56, 23, 128, 36, 126, 32, 69, 66, 115, 59, 29, 19, 88, 28, 125, 7, 10, 94, 99, 77, 17, 51, 21, 75, 67, 87, 90, 102, 41, 14, 120, 89, 33, 108, 2, 86, 64, 106, 78, 116, 24, 61, 101, 40, 82, 85, 97, 63, 92, 119, 74, 60, 13, 68, 30, 72, 8, 103, 50, 133, 37, 112, 0, 114, 109, 91, 4, 12, 43, 118, 9, 139, 11, 25, 100, 38, 136, 27, 110, 6, 98, 18, 113, 42, 117, 123, 105, 76, 73, 129, 49, 65, 22, 111, 70, 79, 45, 137, 134, 81, 52, 93, 83, 71, 15, 138, 130, 34, 39, 20, 48, 1, 104, 54, 80, 26, 53, 16, 127, 107, 44, 121, 31, 131, 35, 62, 84, 47, 46, 58],
        [63, 106, 52, 125, 62, 70, 7, 14, 36, 17, 112, 124, 92, 138, 115, 96, 47, 93, 64, 102, 81, 123, 75, 45, 120, 49, 136, 71, 89, 44, 56, 9, 29, 32, 91, 111, 110, 43, 65, 24, 137, 10, 77, 20, 109, 86, 13, 42, 60, 97, 87, 67, 61, 103, 127, 68, 15, 131, 74, 55, 2, 46, 22, 119, 34, 139, 12, 88, 21, 54, 80, 25, 94, 51, 79, 130, 108, 27, 59, 26, 4, 0, 132, 53, 41, 114, 126, 117, 48, 134, 99, 107, 101, 31, 129, 116, 78, 8, 128, 95, 122, 113, 3, 18, 58, 118, 104, 73, 76, 72, 100, 11, 84, 98, 133, 28, 121, 16, 105, 33, 1, 69, 38, 57, 37, 50, 35, 39, 19, 5, 23, 82, 85, 30, 90, 83, 40, 66, 135, 6],
        [52, 109, 90, 118, 73, 24, 50, 54, 77, 122, 83, 6, 98, 120, 47, 82, 88, 79, 104, 100, 58, 26, 72, 67, 115, 33, 71, 8, 123, 10, 34, 137, 91, 25, 22, 86, 56, 85, 127, 64, 133, 103, 108, 62, 5, 80, 35, 136, 11, 20, 70, 57, 107, 59, 21, 69, 44, 81, 49, 51, 102, 95, 41, 92, 27, 18, 96, 30, 89, 36, 87, 129, 42, 32, 3, 4, 105, 126, 139, 46, 43, 12, 17, 130, 38, 93, 112, 48, 0, 101, 119, 65, 114, 37, 66, 134, 53, 128, 75, 19, 13, 29, 113, 1, 2, 31, 68, 45, 76, 111, 99, 131, 110, 84, 14, 23, 55, 16, 138, 106, 117, 40, 97, 125, 63, 121, 39, 7, 74, 124, 116, 60, 9, 94, 135, 28, 61, 78, 15, 132],
        [34, 103, 95, 33, 7, 138, 54, 38, 29, 78, 71, 110, 73, 70, 112, 81, 68, 56, 69, 117, 129, 89, 27, 105, 14, 1, 25, 104, 9, 24, 53, 59, 102, 3, 49, 50, 97, 61, 135, 20, 37, 128, 122, 86, 60, 8, 87, 39, 2, 83, 106, 75, 127, 35, 21, 80, 120, 132, 17, 93, 45, 0, 64, 28, 63, 52, 19, 18, 108, 48, 125, 44, 107, 57, 12, 90, 62, 5, 111, 32, 11, 99, 30, 136, 137, 94, 139, 100, 67, 40, 36, 66, 26, 22, 116, 47, 101, 113, 43, 115, 82, 42, 41, 114, 55, 58, 16, 77, 134, 96, 65, 109, 98, 79, 92, 124, 85, 91, 118, 88, 46, 121, 72, 131, 76, 23, 130, 10, 13, 84, 15, 119, 4, 126, 133, 51, 74, 6, 31, 123],
        [118, 132, 60, 33, 87, 76, 96, 29, 27, 103, 92, 116, 111, 1, 18, 134, 77, 137, 47, 41, 95, 24, 42, 45, 30, 28, 78, 44, 107, 73, 25, 51, 91, 81, 53, 26, 9, 12, 32, 100, 94, 113, 40, 123, 2, 66, 52, 36, 106, 22, 0, 70, 139, 82, 128, 64, 56, 17, 11, 57, 69, 8, 131, 99, 71, 83, 46, 84, 38, 126, 90, 97, 54, 59, 133, 43, 119, 14, 74, 15, 13, 130, 20, 121, 67, 48, 129, 136, 104, 4, 125, 117, 75, 114, 88, 138, 86, 80, 19, 93, 55, 108, 109, 85, 58, 7, 34, 21, 35, 65, 63, 124, 39, 6, 115, 110, 61, 127, 72, 112, 3, 37, 105, 135, 89, 122, 16, 79, 10, 49, 23, 31, 120, 68, 101, 50, 98, 62, 102, 5]
    ]

    for i in range(num_perm): # range(seed, seed + num_perm)
        # np.random.shuffle(labeled_node_list)  # Randomize order of labeled nodes
        labeled_node_list = permutations[i]
        cur_labeled_node_list = []
        pre_performance = 0
        for lab_idx, labeled_node in enumerate(labeled_node_list):
            for _ in range(wg_l1):  # within group level 1 (for each labeled node)
                # Process 1-hop neighbors
                cur_hop_1_list = []
                hop_1_list = list(labeled_to_player_map[labeled_node].keys())
                np.random.shuffle(hop_1_list)  # Randomize order
                # Keep the precedence constraint between labeled and 1-hop neighbor by putting labeled node front
                hop_1_list.remove(labeled_node)
                hop_1_list.insert(0, labeled_node)

                truncate_length = int(np.ceil((len(hop_1_list) - 1) * (1 - group_trunc_ratio_hop_1))) + 1
                truncate_length = min(truncate_length, len(hop_1_list))
                hop_1_list = hop_1_list[:truncate_length]

                if verbose:
                    print('hop_1_list after truncation', len(hop_1_list))
                    print('labeled_node iteration:', i)
                    print('current target labeled_node:', cur_labeled_node_list, '=>', labeled_node)
                    print('hop_1_list', hop_1_list)

                for player_hop_1 in hop_1_list:
                    cur_hop_1_list.append(player_hop_1)
                    for _ in range(wg_l2):
                        cur_hop_2_list = []
                        hop_2_list = list(labeled_to_player_map[labeled_node][player_hop_1].keys())
                        np.random.shuffle(hop_2_list)  # Randomize order
                        # Keep the precedence constraint between 1-hop neighbor and 2-hop neighbor
                        hop_2_list.remove(player_hop_1)
                        hop_2_list.insert(0, player_hop_1)

                        truncate_length = int(np.ceil((len(hop_2_list) - 1) * (1 - group_trunc_ratio_hop_2))) + 1
                        truncate_length = min(truncate_length, len(hop_2_list))
                        hop_2_list = hop_2_list[:truncate_length]

                        if verbose:
                            print('hop_2_list after truncation', len(hop_2_list))

                        for player_hop_2 in hop_2_list:
                            cur_hop_2_list.append(player_hop_2)
                            # Local propagation and performance computation
                            ind_train_features, ind_train_labels = generate_features_and_labels_ind(
                                cur_hop_1_list, cur_hop_2_list, cur_labeled_node_list, labeled_node,
                                labeled_to_player_map, inductive_edge_index, data, device)
                            assert set(cur_hop_1_list).issubset(hop_1_list)
                            assert set(cur_hop_2_list).issubset(hop_2_list)
                            assert len(cur_labeled_node_list) == lab_idx
                            val_acc = evaluate_retrain_model(
                                MLP, dataset.num_features, dataset.num_classes,
                                ind_train_features, ind_train_labels, val_features, val_labels,
                                device, num_iter=num_epochs, lr=lr, weight_decay=weight_decay)

                            eval_count += 1
                            eval_progress.update(1)
                            if eval_count >= max_model_retrainings:
                                if verbose:
                                    print("Termination condition reached: Maximum evaluations exceeded.")
                                return
                            assert labeled_node == player_hop_1 == player_hop_2 or labeled_node != player_hop_1
                            if labeled_node == player_hop_2 != player_hop_1:
                                assert val_acc - pre_performance < 1e-5, "marginal value found"
                            sample_value_dict[labeled_node][player_hop_1][player_hop_2] += (val_acc - pre_performance)
                            sample_counter_dict[labeled_node][player_hop_1][player_hop_2] += 1
                            pre_performance = val_acc

                            perf_dict["dataset"].append(dataset_name)
                            perf_dict["seed"].append(seed)
                            perf_dict["perm"].append(i)
                            perf_dict["label"].append(labeled_node)
                            perf_dict["first_hop"].append(player_hop_1)
                            perf_dict["second_hop"].append(player_hop_2)
                            perf_dict["accu"].append(val_acc)
                            perf_dict["model_retraining_idx"].append(eval_count)

            cur_labeled_node_list.append(labeled_node)
            ind_train_features = X_ind_propogated[cur_labeled_node_list]
            ind_train_labels = data.y[cur_labeled_node_list]

            val_acc = evaluate_retrain_model(
                MLP, dataset.num_features, dataset.num_classes,
                ind_train_features, ind_train_labels, val_features, val_labels, device)

            eval_count += 1
            eval_progress.update(1)
            if eval_count >= max_model_retrainings:
                if verbose:
                    print("Termination condition reached: Maximum evaluations exceeded.")
                return

            pre_performance = val_acc
            if verbose:
                print('full group acc:', val_acc)

        print(f"Permutation {i} finished, seed {seed}")

    eval_progress.close()


if __name__ == "__main__":

    # Parse command line arguments
    args = parse_args()
    print(args)

    # Set up dataset and model parameters
    dataset_name = args.dataset
    num_hops = args.num_hops
    seed = args.seed
    num_perm = args.num_perm
    label_trunc_ratio = args.label_trunc_ratio
    group_trunc_ratio_hop_1 = args.group_trunc_ratio_hop_1
    group_trunc_ratio_hop_2 = args.group_trunc_ratio_hop_2
    verbose = args.verbose
    wg_l1 = args.wg_l1
    wg_l2 = args.wg_l2
    exp_id = args.exp_id
    # print(f"logs_{dataset_name}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}.txt")
    # sys.stdout = open(f"logs_{dataset_name}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}.txt", "w")
    if dataset_name in dataset_params:
        params = dataset_params[dataset_name]
        num_epochs = params['num_epochs']
        lr = params['lr']
        weight_decay = params['weight_decay']
        print(params)
    else:
        num_epochs = 200
        lr = 0.01
        weight_decay = 5e-4

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load dataset
    if args.dataset in ['Computers', 'Photo']:
        dataset = Amazon(root='dataset/Amazon', name=args.dataset, transform=T.NormalizeFeatures())
        config_path = f'./config/Amazon-{args.dataset}.pkl'
    elif args.dataset == 'Physics':
        dataset = Coauthor(root='dataset/Coauthor', name=args.dataset, transform=T.NormalizeFeatures())
        config_path = f'./config/Coauthor-{args.dataset}.pkl'
    elif args.dataset in ['WikiCS', 'WikiCSX', ]:
        dataset = WikiCS(root='dataset/WikiCS', transform=T.NormalizeFeatures())
        config_path = f'./config/wikics.pkl'
        # generate_wikics_split(dataset)  # if you want to generate the wikics split and save it into a pickle at config dir

    else:
        dataset = Planetoid(root='dataset/' + dataset_name, name=dataset_name, transform=T.NormalizeFeatures())

    data = dataset[0].to(device)
    num_classes = dataset.num_classes

    # Load train/valid/test split for non-Citation datasets
    if args.dataset in ['Computers', 'Photo', 'Physics', 'WikiCS', 'WikiCSX']:
        with open(config_path, 'rb') as f:
            loaded_indices_dict = pickle.load(f)
            if args.dataset in ['WikiCS', 'WikiCSX']:
                assert calculate_md5_of_string(str(loaded_indices_dict)) == "ff62ecc913c95fba03412f445aae153f"
                split_id = loaded_indices_dict["split_id"]
                data.train_mask = data.train_mask[:, split_id].clone()
                data.val_mask = data.val_mask[:, split_id].clone()

                print("before standardize_features", data.x.min(), "to", data.x.max())
                data.x = standardize_features(data.x)
                print("after standardize_features", data.x.min(), "to", data.x.max())

            data = set_masks_from_indices(data, loaded_indices_dict, device)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    if verbose:
        train_size = train_mask.sum().item()
        val_size = val_mask.sum().item()
        test_size = test_mask.sum().item()
        print(f"Train Mask:{train_mask.shape} Size: {train_size}")
        print(f"Validation Mask:{val_mask.shape} Size: {val_size}")
        print(f"Test Mask:{test_mask.shape} Size: {test_size}")

        # Assertions to ensure no overlap between masks
        assert (train_mask & val_mask).sum().item() == 0, "Train and Validation masks overlap!"
        assert (val_mask & test_mask).sum().item() == 0, "Validation and Test masks overlap!"
        assert (train_mask & test_mask).sum().item() == 0, "Train and Test masks overlap!"

    # Prepare validation and test data
    val_edge_index = get_subgraph_data(data.edge_index, val_mask)
    X_val_propogated = propagate_features(val_edge_index, data.x)
    test_edge_index = get_subgraph_data(data.edge_index, test_mask)
    X_test_propogated = propagate_features(test_edge_index, data.x)

    val_features = X_val_propogated[val_mask]
    val_labels = data.y[val_mask]
    test_features = X_test_propogated[test_mask]
    test_labels = data.y[test_mask]

    # Create inductive edge index (removing edges to val/test nodes)
    inductive_edge_index = []
    for src, tgt in data.edge_index.t().tolist():
        if not (val_mask[src] or test_mask[src] or val_mask[tgt] or test_mask[tgt]):
            inductive_edge_index.append([src, tgt])
    inductive_edge_index = torch.tensor(inductive_edge_index).t().contiguous()
    X_ind_propogated = propagate_features(inductive_edge_index, data.x)

    # conv1 = SGConvNoWeight(K=1)
    # X_ind_propogated1 = conv1(data.x, inductive_edge_index)
    #
    # conv2 = SGConvNoWeight(K=2)
    # X_ind_propogated2 = conv2(data.x, inductive_edge_index)
    #
    # conv3 = SGConvNoWeight(K=3)
    # X_ind_propogated3 = conv3(data.x, inductive_edge_index)
    #
    # edge_index2 = k_hop_subgraph(45, num_hops=2, edge_index=inductive_edge_index, relabel_nodes=False)[1]
    # edge_index3 = k_hop_subgraph(45, num_hops=3, edge_index=inductive_edge_index, relabel_nodes=False)[1]
    #
    # X_ind_propogated12 = conv1(data.x, edge_index2)
    # X_ind_propogated22 = conv2(data.x, edge_index2)
    # X_ind_propogated32 = conv3(data.x, edge_index2)
    #
    # X_ind_propogated13 = conv1(data.x, edge_index3)
    # X_ind_propogated23 = conv2(data.x, edge_index3)
    # X_ind_propogated33 = conv3(data.x, edge_index3)
    #
    # print("torch.allclose(X_ind_propogated1[45], X_ind_propogated12[45])", torch.allclose(X_ind_propogated1[45], X_ind_propogated12[45]))
    # print("torch.allclose(X_ind_propogated1[45], X_ind_propogated13[45])", torch.allclose(X_ind_propogated1[45], X_ind_propogated13[45]))
    #
    # print("torch.allclose(X_ind_propogated2[45], X_ind_propogated23[45])", torch.allclose(X_ind_propogated2[45], X_ind_propogated23[45]))
    #
    # time.sleep(2)
    # print()

    if verbose:
        original_edge_count = data.edge_index.size(1)
        inductive_edge_count = inductive_edge_index.size(1)
        print(f"Original Edge Count: {original_edge_count}")
        print(f"Inductive Edge Count: {inductive_edge_count}")

    # Prepare storage data structures for PC-Winter algorithm
    train_idx = torch.nonzero(train_mask).cpu().numpy().flatten()
    labeled_node_list = list(train_idx)
    labeled_to_player_map, sample_value_dict, sample_counter_dict = \
        generate_maps(list(train_idx), num_hops, inductive_edge_index)

    # Store the performance of different seed, permutation index, added new contribution path and accrued performace
    perf_dict = {
        'dataset': [], 'seed': [], 'perm': [], 'label': [],
        'first_hop': [], 'second_hop': [], 'accu': [], "model_retraining_idx": []
    }
    val_acc = evaluate_retrain_model(MLP, dataset.num_features, dataset.num_classes,
                                     X_ind_propogated, data.y, val_features, val_labels,
                                     device, num_iter=num_epochs, lr=lr, weight_decay=weight_decay)

    pc_winter(wg_l1=wg_l1, wg_l2=wg_l2, max_model_retrainings=10000000000000000, verbose=False)
    print("last permutation", perf_dict["perm"][-1])
    #############################################
    # Save results
    os.makedirs(f"value/{exp_id}", exist_ok=True)
    with open(f"value/{exp_id}/{dataset_name}_{wg_l1}_{wg_l2}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value.pkl", "wb") as f:
        pickle.dump(sample_value_dict, f)
    with open(f"value/{exp_id}/{dataset_name}_{wg_l1}_{wg_l2}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_pc_value_count.pkl", "wb") as f:
        pickle.dump(sample_counter_dict, f)
    with open(f"value/{exp_id}/{dataset_name}_{wg_l1}_{wg_l2}_{seed}_{num_perm}_{label_trunc_ratio}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_perf.pkl", "wb") as f:
        pickle.dump(perf_dict, f)
