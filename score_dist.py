import os.path
import pickle
import glob
from collections import Counter
import matplotlib.pyplot as plt
import re


def flatten_dict(d):
    """
    Recursively flatten a 3-level nested dictionary and collect all values.

    Parameters:
    - d: A dictionary of the form d[k1][k2][k3] = v

    Returns:
    - A list of all values v from the nested dictionary.
    """
    flattened_values = []

    for k1 in d:
        for k2 in d[k1]:
            for k3 in d[k1][k2]:
                flattened_values.append(d[k1][k2][k3])

    return flattened_values


def analyze_dist(values, x_axis, title, filename):
    # Use Counter to get the most common values
    value_counts = Counter(values)
    most_common_values = value_counts.most_common(5)

    # Total number of values
    total_values = sum(value_counts.values())

    # Print the top 5 values and their percentages
    print(f"{'Value':<30}{'Count':<30}{'Percentage':<30}")
    for value, count in most_common_values:
        percentage = (count / total_values) * 100
        print(f"{value:<30}{count:<30}{percentage:.2f}%")

    # Plot the distribution
    plt.hist(values, bins=10, alpha=0.75, edgecolor='black')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()


def process_and_combine_files(pattern, x_axis, title_base, file_suffix):
    combined_values = []

    files = glob.glob(pattern)
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            combined_values.extend(flatten_dict(data))

    filename = os.path.join("imgs", f"{title_base}_{file_suffix}.png")
    analyze_dist(combined_values, x_axis, title_base, filename)


# Process Cora files
process_and_combine_files("value/Cora_*(\d+)_10_0_0.5_0.7_pc_value.pkl",
                          "PC Value",
                          "Combined Distribution of Cora Values",
                          "pc_value")
process_and_combine_files("value/Cora_*(\d+)_10_0_0.5_0.7_pc_value_count.pkl",
                          "Node updates during pc-value-eval",
                          "Combined Distribution of Cora Counts",
                          "pc_value_count")

# Process WikiCS files #
process_and_combine_files("value/WikiCS_*(\d+)_1_0_0.7_0.9_pc_value.pkl",
                          "PC Value",
                          "Combined Distribution of WikiCS Values", "pc_value")
process_and_combine_files("value/WikiCS_*(\d+)_1_0_0.7_0.9_pc_value_count.pkl",
                          "Node updates during pc-value-eval",
                          "Combined Distribution of WikiCS Counts",
                          "pc_value_count")
