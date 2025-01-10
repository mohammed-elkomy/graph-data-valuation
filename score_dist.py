import os
import pickle
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def aggregate_data(file_list, is_count):
    """
    Aggregates data from a list of files.

    Parameters:
    - file_list: List of filenames to process.
    - is_count: Boolean indicating whether the files contain counts (True) or values (False).

    Returns:
    - A dictionary of aggregated data.
    """
    results = defaultdict(list) if not is_count else defaultdict(int)

    for filename in file_list:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            for key, sub_dict in data.items():
                for sub_key, sub_sub_dict in sub_dict.items():
                    for sub_sub_key, value in sub_sub_dict.items():
                        if is_count:
                            results[key] += value
                            results[sub_key] += value
                            results[sub_sub_key] += value
                        else:
                            results[(key, sub_key, sub_sub_key)].append(value)

    return results


def analyze_dist(values, x_axis, title, filename):
    """
    Analyzes and plots the distribution of values.

    Parameters:
    - values: List of values to analyze.
    - x_axis: Label for the x-axis of the plot.
    - title: Title of the plot.
    - filename: Filename to save the plot.
    """
    value_counts = Counter(values)
    most_common_values = value_counts.most_common(15)

    total_values = sum(value_counts.values())

    print(f"Analyzing: {filename}")
    print(f"{'Value':<30}{'Count':<30}{'Percentage':<30}")
    for value, count in most_common_values:
        percentage = (count / total_values) * 100
        print(f"{value:<30}{count:<30}{percentage:.2f}%")
    print()

    plt.hist(values, bins=100, alpha=0.75, edgecolor='black')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def process_and_combine_files(pattern, x_axis, title_base, file_suffix, is_count=False, num_perms=1):
    files = glob.glob(pattern)
    combined_data = aggregate_data(files, is_count)

    if not is_count:
        # For value files, calculate the average values
        combined_values = [sum(values) / (len(values) * num_perms) for values in combined_data.values()]
    else:
        # For count files, the combined_data already contains the aggregated counts
        combined_values = list(combined_data.values())

    print(f"Loaded files: {files}")
    filename = os.path.join("imgs", f"{title_base}_{file_suffix}.png")
    analyze_dist(combined_values, x_axis, title_base, filename)


# Define file patterns and process each
process_and_combine_files(r"value/Cora_*_10_0_0.5_0.7_pc_value.pkl",
                          "PC Value",
                          "Combined Distribution of Cora Values",
                          "pc_value",
                          is_count=False,
                          num_perms=10)

process_and_combine_files(r"value/Cora_*_10_0_0.5_0.7_pc_value_count.pkl",
                          "Node updates during pc-value-eval",
                          "Combined Distribution of Cora Counts",
                          "pc_value_count",
                          is_count=True)

process_and_combine_files(r"value/WikiCS_*_1_0_0.7_0.9_pc_value.pkl",
                          "PC Value",
                          "Combined Distribution of WikiCS Values",
                          "pc_value",
                          is_count=False,
                          num_perms=1)

process_and_combine_files(r"value/WikiCS_*_1_0_0.7_0.9_pc_value_count.pkl",
                          "Node updates during pc-value-eval",
                          "Combined Distribution of WikiCS Counts",
                          "pc_value_count",
                          is_count=True)
