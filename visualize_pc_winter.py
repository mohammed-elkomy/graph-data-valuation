import glob
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt

img_dir = "imgs"

dataset = "Cora"
group_trunc_ratio_hop_1 = 0.5
group_trunc_ratio_hop_2 = 0.7

pattern = f'./res/*node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_*_{dataset}_test.pkl'

datagroups = defaultdict(list)
# Get the unique part of the filename based on the wildcard `*`
for file_path in sorted(glob.glob(pattern)):
    print("processing subset", file_path)
    # Extract the unique part of the filename based on the wildcard `*`
    base_name = os.path.basename(file_path)
    unique_part = base_name.split('_')[8]  # Assuming the unique part is after the 6th underscore

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(len(data))
    datagroups[unique_part].extend(data)

for identifier, data_group in datagroups.items():
    print(len(data_group))
    plt.figure(figsize=(8, 6))

    plt.plot(data_group, label='Our method')

    # Set the x and y axis labels with increased font size
    plt.xlabel('Number of Unlabled Node Removed', fontsize=16)
    plt.ylabel('Prediction Accuracy (%)', fontsize=16)
    plt.title(f'{dataset} up to {identifier}', fontsize=16)

    # Increase the size of the tick labels for both axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Show legend
    plt.legend()

    # Save the figure with the unique part as the filename
    image_path = os.path.join(img_dir, f'drop_up_to_{identifier}.png')
    plt.savefig(image_path)

    # Close the plot to free memory
    plt.close()

    print(f"Saved plot to {image_path}")
