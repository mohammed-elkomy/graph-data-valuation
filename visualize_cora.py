import glob
import os
import pickle
import matplotlib.pyplot as plt

img_dir = "imgs"

pattern = './res/node_drop_large_winter_value_0.5_0.7_*_cora_test.pkl'
for file_path in glob.glob(pattern):
    # Extract the unique part of the filename based on the wildcard `*`
    base_name = os.path.basename(file_path)
    unique_part = base_name.split('_')[7]  # Assuming the unique part is after the 6th underscore

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    plt.figure(figsize=(8, 6))

    plt.plot(data, label='Our method')

    # Set the x and y axis labels with increased font size
    plt.xlabel('Number of Unlabled Node Removed', fontsize=16)
    plt.ylabel('Prediction Accuracy (%)', fontsize=16)
    plt.title(f'Cora up to {unique_part}', fontsize=16)

    # Increase the size of the tick labels for both axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Show legend
    plt.legend()

    # Save the figure with the unique part as the filename
    image_path = os.path.join(img_dir, f'drop_up_to_{unique_part}.png')
    plt.savefig(image_path)

    # Close the plot to free memory
    plt.close()

    print(f"Saved plot to {image_path}")
