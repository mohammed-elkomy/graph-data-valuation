import pickle
import matplotlib.pyplot as plt


file_path = './res/node_drop_large_winter_value_0.5_0.7_2_cora_test.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

plt.figure(figsize=(8, 6))

plt.plot(data, label='Our method')

# Set the x and y axis labels with increased font size
plt.xlabel('Number of Unlabled Node Removed', fontsize=16)
plt.ylabel('Prediction Accuracy (%)', fontsize=16)
plt.title('Cora', fontsize=16)

# Increase the size of the tick labels for both axes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show legend
plt.legend()

# Displaying the plot
plt.show()
