import glob
import os
import pickle
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def nested_aggregation(file_list, is_count, level=None):
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
                        unique_keys = {key, sub_key, sub_sub_key}
                        if level is None or level == len(unique_keys):
                            for k in unique_keys:
                                if is_count:
                                    results[k] += value
                                else:
                                    results[k].append(value)

                            # results[(key, sub_key, sub_sub_key)].append(value)

    return results


def analyze_dist(values, x_axis, y_axis, title, filename, bins, fit_normal=False,
                 legend_prefix="", hist_color="red", fit_color="blue"):
    """
    Analyzes and plots the distribution of values, focusing on the range covering 97% of the data.
    Optionally fits a normal distribution to the histogram bars and shows its mean and sigma.

    Parameters:
    - values: List of values to analyze.
    - x_axis: Label for the x-axis of the plot.
    - y_axis: Label for the y-axis of the plot.
    - title: Title of the plot.
    - filename: Filename to save the plot.
    - bins: Number of bins in the histogram.
    - fit_normal: Whether to fit a normal distribution to the histogram bars and display its mean and sigma.
    - legend_prefix: Prefix for the legend labels to differentiate multiple calls.
    - hist_color: Color of the histogram.
    - fit_color: Color of the normal distribution fit.
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

    # Plot the histogram with filtered values
    n, bins, patches = plt.hist(filtered_values, bins=bins, edgecolor=hist_color, color=hist_color, alpha=0.4,
                                label=f"{legend_prefix} Histogram")

    # Set y-ticks to show 10 evenly spaced ticks
    y_max = n.max()  # Maximum density value from the histogram
    y_ticks = np.linspace(0, y_max, 10)  # 10 evenly spaced ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y / len(values) * 100:.2f}%'))
    plt.yticks(y_ticks, [f'{y / len(values) * 100:.2f}%' for y in y_ticks])

    # Adjust the x-axis limits based on the percentile bounds
    plt.xlim(left=lower_bound * 0.9 - 1e-4, right=upper_bound * 1.1 + 1e-4)

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
        plt.plot(x, y, color=fit_color, linestyle='-',
                 label=f'{legend_prefix} Normal Distribution\nMean = {mean:.5e}, Sigma = {std:.5e}')

    plt.legend()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)


def process_and_combine_files_values(pattern, x_axis, y_axis, title_base, file_suffix, fit_normal, num_perms, legend_prefix, hist_color, fit_color):
    """
    Process and combine files when is_count is False. It handles the aggregation of values (non-count data).

    Parameters:
    - pattern: File pattern to match the files.
    - x_axis: Label for the x-axis of the plot.
    - y_axis: Label for the y-axis of the plot.
    - title_base: Base title for the plot.
    - file_suffix: Suffix to be added to the plot filename.
    - fit_normal: Boolean indicating whether to fit a normal distribution.
    - num_perms: Number of permutations to calculate the average.
    """
    print(legend_prefix, hist_color, fit_color)
    files = glob.glob(pattern)
    combined_data = nested_aggregation(files, is_count=False)

    # Calculate the average values for each key in the combined data
    combined_values = [sum(values) / (len(values) * num_perms) for values in combined_data.values()]

    print(f"XXXLoaded files: {files}")
    filename = os.path.join("imgs", f"{title_base}_{file_suffix}.png")
    title_base = "Cora within group analysis"
    analyze_dist(combined_values, x_axis, y_axis, title_base, filename, fit_normal=fit_normal, bins=100, legend_prefix=legend_prefix, hist_color=hist_color, fit_color=fit_color)


def process_and_combine_files_values_trunc_counts(pattern, x_axis, y_axis, title_base, file_suffix, fit_normal, num_perms):
    """
    Process and combine value_files when is_count is False. It handles the aggregation of values (non-count data).

    Parameters:
    - pattern: File pattern to match the value_files.
    - x_axis: Label for the x-axis of the plot.
    - y_axis: Label for the y-axis of the plot.
    - title_base: Base title for the plot.
    - file_suffix: Suffix to be added to the plot filename.
    - fit_normal: Boolean indicating whether to fit a normal distribution.
    - num_perms: Number of permutations to calculate the average.
    """
    value_files = glob.glob(pattern)
    count_files = glob.glob(pattern.replace("pc_value.pkl", "pc_value_count.pkl"))
    val_agg_data = nested_aggregation(value_files, is_count=False)
    count_agg_data = nested_aggregation(count_files, is_count=True)

    p = np.percentile(np.array(list(count_agg_data.values())), 70)
    print("p", p)
    well_represented_nodes = [k for k, v in count_agg_data.items() if v > p]
    # Calculate the average values for each key in the combined data
    combined_values = [sum(values) / (len(values) * num_perms)
                       for node_id, values in val_agg_data.items()
                       if node_id in well_represented_nodes]

    print(f"Loaded value_files: {value_files}")
    filename = os.path.join("imgs", f"{title_base}_{file_suffix}.png")
    analyze_dist(combined_values, x_axis, y_axis, title_base, filename, bins=100, fit_normal=fit_normal)


def process_and_combine_files_counts(pattern, x_axis, y_axis, title_base, file_suffix, fit_normal, num_perms):
    """
    Process and combine files when is_count is True. It handles the aggregation of counts (count data).

    Parameters:
    - pattern: File pattern to match the files.
    - x_axis: Label for the x-axis of the plot.
    - y_axis: Label for the y-axis of the plot.
    - title_base: Base title for the plot.
    - file_suffix: Suffix to be added to the plot filename.
    - fit_normal: Boolean indicating whether to fit a normal distribution.
    - num_perms: Number of permutations to calculate the average (if necessary).
    """
    files = glob.glob(pattern)
    combined_data = nested_aggregation(files, is_count=True)

    # The combined_data already contains the aggregated counts
    combined_values = list(combined_data.values())

    print(f"Loaded files: {files}")
    filename = os.path.join("imgs", f"{title_base}_{file_suffix}.png")
    analyze_dist(combined_values, x_axis, y_axis, title_base, filename, fit_normal=fit_normal, bins=10000)


def per_level_analyze_dist(pattern):
    # x_axis, y_axis, title_base, file_suffix, fit_normal, num_perms, legend_prefix, hist_color, fit_color
    files = glob.glob(pattern)
    hop0 = nested_aggregation(files, is_count=False, level=1)
    hop1 = nested_aggregation(files, is_count=False, level=2)
    hop2 = nested_aggregation(files, is_count=False, level=3)
    hop0_pcw_values = sum(hop0.values(), [])
    hop1_pcw_values = sum(hop1.values(), [])
    hop2_pcw_values = sum(hop2.values(), [])

    # Compute variance
    var1 = np.var(hop0_pcw_values)
    var2 = np.var(hop1_pcw_values)
    var3 = np.var(hop2_pcw_values)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist([hop0_pcw_values, hop1_pcw_values, hop2_pcw_values], bins=200, alpha=0.7, density=True, label=[
        f'hop 0 (Var: {var1:.5f})',
        f'hop 1 (Var: {var2:.5f})',
        f'hop 2 (Var: {var3:.5f})'
    ])

    # Labels and title
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Normalized Histogram Distribution with Variance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([-.5, .5])

    plt.tight_layout()
    plt.savefig(os.path.join("imgs", "per-level.png"))


per_level_analyze_dist(r"value/fixed-perm/*_pc_value.pkl", )

plt.figure(figsize=(14, 7))

process_and_combine_files_values(r"value/same model retrainings/Cora_1_5_*_50_0_0.5_0.7_pc_value.pkl",
                                 "PC Value", "Percentage of nodes",
                                 "Combined Distribution of Cora Values 1 5",
                                 "pc_value",
                                 fit_normal=True,
                                 num_perms=1, legend_prefix="(1-5)", hist_color="cyan", fit_color="cyan")

process_and_combine_files_values(r"value/same model retrainings/Cora_1_10_*_50_0_0.5_0.7_pc_value.pkl",
                                 "PC Value", "Percentage of nodes",
                                 "Combined Distribution of Cora Values 1 10",
                                 "pc_value",
                                 fit_normal=True,
                                 num_perms=1, legend_prefix="(1-10)", hist_color="yellow", fit_color="yellow")

process_and_combine_files_values(r"value/same model retrainings/Cora_5_10_*_50_0_0.5_0.7_pc_value.pkl",
                                 "PC Value", "Percentage of nodes",
                                 "Combined Distribution of Cora Values 5 10",
                                 "pc_value",
                                 fit_normal=True,
                                 num_perms=1, legend_prefix="(5-10)", hist_color="red", fit_color="red")

process_and_combine_files_values(r"value/same model retrainings/Cora_5_1_*_50_0_0.5_0.7_pc_value.pkl",
                                 "PC Value", "Percentage of nodes",
                                 "Combined Distribution of Cora Values 5 1",
                                 "pc_value",
                                 fit_normal=True,
                                 num_perms=1, legend_prefix="(5-1)", hist_color="green", fit_color="green")

process_and_combine_files_values(r"value/same model retrainings/Cora_10_1_*_50_0_0.5_0.7_pc_value.pkl",
                                 "PC Value", "Percentage of nodes",
                                 "Combined Distribution of Cora Values 10 1",
                                 "pc_value",
                                 fit_normal=True,
                                 num_perms=1, legend_prefix="(10-1)", hist_color="purple", fit_color="purple")

process_and_combine_files_values(r"value/same model retrainings/Cora_1_1_*_50_0_0.5_0.7_pc_value.pkl",
                                 "PC Value", "Percentage of nodes",
                                 "Combined Distribution of Cora Values 1 1",
                                 "pc_value",
                                 fit_normal=True,
                                 num_perms=1, legend_prefix="(1-1)", hist_color="blue", fit_color="blue")  # ignored normalize factor
plt.close()

plt.figure(figsize=(14, 7))
process_and_combine_files_counts(r"value/same model retrainings/Cora_1_1_*_50_0_0.5_0.7_pc_value_count.pkl",
                                 "Node updates during pc-winter value evaluation", "Percentage of nodes",
                                 "Combined Distribution of Cora Counts 1 1",
                                 "pc_value_count",
                                 fit_normal=False,
                                 num_perms=1)
plt.close()
plt.figure(figsize=(14, 7))
process_and_combine_files_counts(r"value/same model retrainings/Cora_5_10_*_50_0_0.5_0.7_pc_value_count.pkl",
                                 "Node updates during pc-winter value evaluation", "Percentage of nodes",
                                 "Combined Distribution of Cora Counts 5 10",
                                 "pc_value_count",
                                 fit_normal=False,
                                 num_perms=1)
plt.close()
plt.figure(figsize=(14, 7))
process_and_combine_files_counts(r"value/same model retrainings/Cora_5_1_*_50_0_0.5_0.7_pc_value_count.pkl",
                                 "Node updates during pc-winter value evaluation", "Percentage of nodes",
                                 "Combined Distribution of Cora Counts 5 1",
                                 "pc_value_count",
                                 fit_normal=False,
                                 num_perms=1)
plt.close()
plt.figure(figsize=(14, 7))
process_and_combine_files_counts(r"value/same model retrainings/Cora_10_1_*_50_0_0.5_0.7_pc_value_count.pkl",
                                 "Node updates during pc-winter value evaluation", "Percentage of nodes",
                                 "Combined Distribution of Cora Counts 10 1",
                                 "pc_value_count",
                                 fit_normal=False,
                                 num_perms=1)
plt.close()
plt.figure(figsize=(14, 7))

# process_and_combine_files_values(r"value/Cora_*_10_0_0.5_0.7_pc_value.pkl",
#                                  "PC Value", "Percentage of nodes",
#                                  "Combined Distribution of Cora Values",
#                                  "pc_value",
#                                  fit_normal=True,
#                                  num_perms=10)
#
# process_and_combine_files_counts(r"value/Cora_*_10_0_0.5_0.7_pc_value_count.pkl",
#                                  "Node updates during pc-winter value evaluation", "Percentage of nodes",
#                                  "Combined Distribution of Cora Counts",
#                                  "pc_value_count",
#                                  fit_normal=False,
#                                  num_perms=10)
#
# process_and_combine_files_values(r"value/PubMed_*_10_0_0.5_0.7_pc_value.pkl",
#                                  "PC Value", "Percentage of nodes",
#                                  "Combined Distribution of PubMed Values",
#                                  "pc_value",
#                                  fit_normal=True,
#                                  num_perms=10)
#
# process_and_combine_files_counts(r"value/PubMed_*_10_0_0.5_0.7_pc_value_count.pkl",
#                                  "Node updates during pc-winter value evaluation", "Percentage of nodes",
#                                  "Combined Distribution of PubMed Counts",
#                                  "pc_value_count",
#                                  fit_normal=False,
#                                  num_perms=10)
#
# process_and_combine_files_values(r"value/CiteSeer_*_10_0_0.5_0.7_pc_value.pkl",
#                                  "PC Value", "Percentage of nodes",
#                                  "Combined Distribution of CiteSeer Values",
#                                  "pc_value",
#                                  fit_normal=True,
#                                  num_perms=10)
#
# process_and_combine_files_counts(r"value/CiteSeer_*_10_0_0.5_0.7_pc_value_count.pkl",
#                                  "Node updates during pc-winter value evaluation", "Percentage of nodes",
#                                  "Combined Distribution of CiteSeer Counts",
#                                  "pc_value_count",
#                                  fit_normal=False,
#                                  num_perms=10)
#
#
#
# process_and_combine_files_values_trunc_counts(r"value/WikiCS_*_1_0_0.7_0.9_pc_value.pkl",
#                                               "PC Value", "Percentage of nodes",
#                                               "Combined Distribution of WikiCS Values After Truncation",
#                                               "pc_value_truncated",
#                                               fit_normal=True,
#                                               num_perms=1)
#
# process_and_combine_files_values(r"value/WikiCS_*_1_0_0.7_0.9_pc_value.pkl",
#                                  "PC Value", "Percentage of nodes",
#                                  "Combined Distribution of WikiCS Values",
#                                  "pc_value",
#                                  fit_normal=True,
#                                  num_perms=1)
#
# process_and_combine_files_counts(r"value/WikiCS_*_1_0_0.7_0.9_pc_value_count.pkl",
#                                  "Node updates during pc-winter value evaluation", "Percentage of nodes",
#                                  "Combined Distribution of WikiCS Counts",
#                                  "pc_value_count",
#                                  fit_normal=False,
#                                  num_perms=1
#                                  )
#
# process_and_combine_files_values_trunc_counts(r"value/WikiCS2_*_1_0_0.7_0.9_pc_value.pkl",
#                                               "PC Value", "Percentage of nodes",
#                                               "Combined Distribution of WikiCS2 Values After Truncation",
#                                               "pc_value_truncated",
#                                               fit_normal=True,
#                                               num_perms=1)
#
# process_and_combine_files_values(r"value/WikiCS2_*_1_0_0.7_0.9_pc_value.pkl",
#                                  "PC Value", "Percentage of nodes",
#                                  "Combined Distribution of WikiCS2 Values",
#                                  "pc_value",
#                                  fit_normal=True,
#                                  num_perms=1)
#
# process_and_combine_files_counts(r"value/WikiCS2_*_1_0_0.7_0.9_pc_value_count.pkl",
#                                  "Node updates during pc-winter value evaluation", "Percentage of nodes",
#                                  "Combined Distribution of WikiCS2 Counts",
#                                  "pc_value_count",
#                                  fit_normal=False,
#                                  num_perms=1
#                                  )
#
#
# process_and_combine_files_values_trunc_counts(r"value/WikiCSX_*_1_0_0.7_0.9_pc_value.pkl",
#                                               "PC Value", "Percentage of nodes",
#                                               "Combined Distribution of WikiCSX Values After Truncation",
#                                               "pc_value_truncated",
#                                               fit_normal=True,
#                                               num_perms=1)
#
# process_and_combine_files_values(r"value/WikiCSX_*_1_0_0.7_0.9_pc_value.pkl",
#                                  "PC Value", "Percentage of nodes",
#                                  "Combined Distribution of WikiCSX Values",
#                                  "pc_value",
#                                  fit_normal=True,
#                                  num_perms=1)
#
# process_and_combine_files_counts(r"value/WikiCSX_*_1_0_0.7_0.9_pc_value_count.pkl",
#                                  "Node updates during pc-winter value evaluation", "Percentage of nodes",
#                                  "Combined Distribution of WikiCSX Counts",
#                                  "pc_value_count",
#                                  fit_normal=False,
#                                  num_perms=1
#                                  )
