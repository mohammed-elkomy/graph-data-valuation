import glob
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset arguments.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--group_trunc_ratio_hop_1', type=float, required=True, help="Truncation ratio for hop 1")
    parser.add_argument('--group_trunc_ratio_hop_2', type=float, required=True, help="Truncation ratio for hop 2")
    parser.add_argument('--permutation_count', type=int, required=True, help="Number of permutations")
    parser.add_argument('--wg_l1', type=int, required=True,
                        help="count for level 1 within groups")
    parser.add_argument('--wg_l2', type=int, required=True,
                        help="count for level 2 within groups")
    args = parser.parse_args()

    # Setting the variables based on the parsed arguments
    dataset = args.dataset
    group_trunc_ratio_hop_1 = args.group_trunc_ratio_hop_1
    group_trunc_ratio_hop_2 = args.group_trunc_ratio_hop_2
    permutation_count = args.permutation_count

    return dataset, group_trunc_ratio_hop_1, group_trunc_ratio_hop_2, permutation_count


dataset, group_trunc_ratio_hop_1, group_trunc_ratio_hop_2, permutation_count = parse_args()
print(f"Dataset: {dataset}")
print(f"Group Truncation Ratio Hop 1: {group_trunc_ratio_hop_1}")
print(f"Group Truncation Ratio Hop 2: {group_trunc_ratio_hop_2}")
print(f"Permutation Count: {permutation_count}")

img_dir = "imgs"
pattern = f'./res/*node_drop_large_winter_value_{wg_l1}_{wg_l2}_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_*_{dataset}_test.pkl'

datagroups = defaultdict(list)
files_per_dump_count = defaultdict(list)
# Get the unique part of the filename based on the wildcard `*`
for file_path in sorted(glob.glob(pattern)):
    print("processing subset", file_path)
    # Extract the unique part of the filename based on the wildcard `*`
    base_name = os.path.basename(file_path)
    dump_count = base_name.split('_')[7]  # Assuming the dump_count is after the 6th underscore

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    datagroups[dump_count].extend(data)
    files_per_dump_count[dump_count].append(file_path)

for dump_count, matched_files in files_per_dump_count.items():
    sub = "\n".join(matched_files)
    assert len(matched_files) in [10, 1], f"failed to get parts of \n{sub}"
    if len(matched_files) == 1:
        print("only found",sub)

# Function to calculate the simple moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


for dump_count, data_group in datagroups.items():
    dump_count = int(dump_count)
    print(dump_count, len(data_group))
    plt.figure(figsize=(8, 6))

    # Plot the original data
    plt.plot(data_group, label='PC-Winter Value Reproduction')

    window_size = 20
    # Calculate and plot the moving average
    smooth_data = moving_average(data_group, window_size=window_size)
    plt.plot(range(len(data_group) - len(smooth_data) + 1, len(data_group) + 1),
             smooth_data,
             color='red',
             label=f'Moving Average (Window={window_size})')

    # Set the x and y axis labels with increased font size
    plt.xlabel('Number of Unlabeled Nodes Removed', fontsize=16)
    plt.ylabel('Prediction Accuracy (%)', fontsize=16)
    plt.title(f'{dataset} up to {dump_count * permutation_count}', fontsize=16)

    # Increase the size of the tick labels for both axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Show legend
    plt.legend()

    # Save the figure with the unique part as the filename
    image_path = os.path.join(img_dir, f'{dataset}-drop_up_to_{dump_count * permutation_count}.png')
    plt.savefig(image_path)

    # Close the plot to free memory
    plt.close()

    print(f"Saved plot to {image_path}")
