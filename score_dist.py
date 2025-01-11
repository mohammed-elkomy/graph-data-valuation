import glob
import os
import pickle
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats


def analyze_dist(values, x_axis, y_axis, title, filename, fit_normal=False):
    """
    Analyzes and plots the distribution of values, focusing on the range covering 97% of the data.
    Optionally fits a normal distribution to the histogram bars and shows its mean and sigma.

    Parameters:
    - values: List of values to analyze.
    - x_axis: Label for the x-axis of the plot.
    - y_axis: Label for the y-axis of the plot.
    - title: Title of the plot.
    - filename: Filename to save the plot.
    - fit_normal: Whether to fit a normal distribution to the histogram bars and display its mean and sigma.
    """
    value_counts = Counter(values)
    most_common_values = value_counts.most_common(15)

    total_values = sum(value_counts.values())

    print(f"Analyzing: {filename}")
    print(f"{'Value':<30}{'Count':<30}{'Percentage':<30}")
    for value, count in most_common_values:
        percentage = (count / total_values) * 100
        print(f"{value:<30}{count:<30}{percentage:.2f}%")

    # Determine the value at the 97th percentile
    upper_bound = np.percentile(values, 97)
    lower_bound = np.percentile(values, 3)

    # Filter values within the 3rd to 97th percentile
    filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
    plt.figure(figsize=(14, 7))

    # Plot the histogram with filtered values
    n, bins, patches = plt.hist(filtered_values, bins=10000, edgecolor='red', color="red", alpha=0.75)

    # Set y-ticks to show 10 evenly spaced ticks
    y_max = n.max()  # Maximum density value from the histogram
    y_ticks = np.linspace(0, y_max, 10)  # 10 evenly spaced ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y / len(values) * 100:.2f}%'))
    plt.yticks(y_ticks, [f'{y / len(values) * 100:.2f}%' for y in y_ticks])

    # Adjust the x-axis limits based on the percentile bounds
    plt.xlim(left=lower_bound * 0.9 - 1e-3, right=upper_bound * 1.1 + 1e-3)

    # If fitting a normal distribution to the histogram bars
    if fit_normal:
        # Fit a normal distribution to the data (mean and std)
        mean, std = np.mean(filtered_values), np.std(filtered_values)

        # Display the fitted normal distribution's mean and sigma
        print(f"Fitted Normal Distribution: Mean = {mean:.5f}, Sigma = {std:.5f}")

        # Plot the fitted normal distribution over the histogram
        x = np.linspace(lower_bound, upper_bound, 1000)
        y = stats.norm.pdf(x, mean, std)
        y = y / y.max() * y_max
        plt.plot(x, y, 'b-', label=f'Normal Distribution\nMean = {mean:.5f}, Sigma = {std:.5f}')
        plt.legend()

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()


def process_and_combine_files(pattern, x_axis, y_axis, title_base, file_suffix, fit_normal, num_perms, is_count=False):
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
    analyze_dist(combined_values, x_axis, y_axis, title_base, filename, fit_normal=fit_normal)


# Define file patterns and process each
process_and_combine_files(r"value/Cora_*_10_0_0.5_0.7_pc_value.pkl",
                          "PC Value", "Percentage of nodes",
                          "Combined Distribution of Cora Values",
                          "pc_value",
                          fit_normal=True,
                          is_count=False,
                          num_perms=10)

process_and_combine_files(r"value/Cora_*_10_0_0.5_0.7_pc_value_count.pkl",
                          "Node updates during pc-winter value evaluation", "Percentage of nodes",
                          "Combined Distribution of Cora Counts",
                          "pc_value_count",
                          fit_normal=False,
                          is_count=True,
                          num_perms=10

                          )

process_and_combine_files(r"value/WikiCS_*_1_0_0.7_0.9_pc_value.pkl",
                          "PC Value", "Percentage of nodes",
                          "Combined Distribution of WikiCS Values",
                          "pc_value",
                          fit_normal=True,
                          is_count=False,
                          num_perms=1)

process_and_combine_files(r"value/WikiCS_*_1_0_0.7_0.9_pc_value_count.pkl",
                          "Node updates during pc-winter value evaluation", "Percentage of nodes",
                          "Combined Distribution of WikiCS Counts",
                          "pc_value_count",
                          fit_normal=False,
                          is_count=True,
                          num_perms=1
                          )
