import glob
import os
import pickle
import matplotlib.pyplot as plt

img_dir = "imgs"

dataset = "Cora"
group_trunc_ratio_hop_1 = 0.5
group_trunc_ratio_hop_2 = 0.7

pattern = f'./res/*node_drop_large_winter_value_{group_trunc_ratio_hop_1}_{group_trunc_ratio_hop_2}_*_{dataset}_test.pkl'
print(sorted(glob.glob(pattern)))

# Get the unique part of the filename based on the wildcard `*`
for file_path in glob.glob(pattern):
    print(file_path)
    # Extract the unique part of the filename based on the wildcard `*`
    base_name = os.path.basename(file_path)
    unique_part = base_name.split('_')[7]  # Assuming the unique part is after the 6th underscore

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(data, len(data))
    # exit()
    # plt.figure(figsize=(8, 6))
    #
    # plt.plot(data, label='Our method')
    #
    # # Set the x and y axis labels with increased font size
    # plt.xlabel('Number of Unlabled Node Removed', fontsize=16)
    # plt.ylabel('Prediction Accuracy (%)', fontsize=16)
    # plt.title(f'{dataset} up to {unique_part}', fontsize=16)
    #
    # # Increase the size of the tick labels for both axes
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    #
    # # Show legend
    # plt.legend()
    #
    # # Save the figure with the unique part as the filename
    # image_path = os.path.join(img_dir, f'drop_up_to_{unique_part}.png')
    # plt.savefig(image_path)
    #
    # # Close the plot to free memory
    # plt.close()
    #
    # print(f"Saved plot to {image_path}")
