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
    parser.add_argument('--count', type=int, required=True, help="Number of permutations or the number of sets for figure title")
    args = parser.parse_args()

    # Setting the variables based on the parsed arguments
    dataset = args.dataset
    permutation_count = args.count

    return dataset, permutation_count


# Function to calculate the simple moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def process_files(dataset, count):
    img_dir = "imgs"
    os.makedirs(img_dir, exist_ok=True)

    # Glob pattern for all relevant pkl files
    pattern = os.path.join(".", "**", f"*node_drop_large_winter_value_*_{dataset}_test.pkl")
    file_paths = sorted(glob.glob(pattern, recursive=True))

    # Group files based on run_config
    config_groups = defaultdict(list)

    for file_path in file_paths:
        base_name = os.path.basename(file_path)
        root_dir = os.path.split(file_path)[0]
        parts = base_name.split('_')
        run_config = (root_dir, parts[0].split("-")[0],) + tuple(file_path.split("_")[5:9])  # model config and dir
        dump_count = parts[9]  # Extract dump count

        config_groups[run_config].append((dump_count, file_path))

    datagroups = defaultdict(list)
    for run_config, files in config_groups.items():
        for dump_count, file_path in files:
            print(f"Processing subset {file_path}")
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            datagroups[(run_config, dump_count)].extend(data)

    # Generate plots for each grouped dataset
    for (run_config, dump_count), data_group in datagroups.items():
        dump_count = int(dump_count)
        print(f"{run_config}-{dump_count}: {len(data_group)} entries")

        plt.figure(figsize=(8, 6))
        plt.plot(data_group, label='PC-Winter Value Reproduction')

        # Calculate and plot the moving average
        smooth_data = moving_average(data_group, window_size=20)
        plt.plot(range(len(data_group) - len(smooth_data) + 1, len(data_group) + 1),
                 smooth_data,
                 color='red',
                 label='Moving Average (Window=20)')

        # Labels and titles
        plt.xlabel('Number of Unlabeled Nodes Removed', fontsize=16)
        plt.ylabel('Prediction Accuracy (%)', fontsize=16)
        wg_l1, wg_l2 = run_config[2], run_config[3]

        plt.title(f'{dataset} up to {count} ({wg_l1}-{wg_l2})', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()

        # Save figure in structured directories
        prefix_dir = os.path.join(img_dir, run_config[0])
        os.makedirs(prefix_dir, exist_ok=True)

        image_path = os.path.join(prefix_dir, f'{dataset}-drop_up_to_{count}_{wg_l1}_{wg_l2}.png')
        plt.savefig(image_path)
        plt.close()

        print(f"Saved plot to {image_path}")

    # Group data by the first key
    grouped_runs_compare = {}
    for key, values in datagroups.items():
        group_name = key[0][0]  # Extract group name
        # grouped_runs_compare.setdefault(group_name, []).append((key[0][2:]+("B"+key[1],), values)) # TODO
        grouped_runs_compare.setdefault(group_name, []).append((key[0][2:], values))

    # Plot each group
    for group, entries in grouped_runs_compare.items():
        entries = [e for e in entries if int(e[0][1]) == 1]
        plt.figure(figsize=(10, 6))
        for key, values in entries:
            # label = "-".join(key[:2]+key[-1:])  # Create label using last four keys # TODO
            label = "-".join(key[:2])  # Create label using last four keys
            plt.plot(values, linestyle='-', label=label)
        plt.xlabel("Number of Unlabeled Nodes Removed")
        plt.ylabel("Prediction Accuracy (%)")
        plt.title(f"Performance Trends for {os.path.split(group)[-1]}")
        plt.legend()
        plt.grid(True)
        filename = os.path.join(img_dir, group, "overall.png")
        plt.savefig(filename)


if __name__ == "__main__":
    dataset, permutation_count = parse_args()
    process_files(dataset, permutation_count)
