import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap


plt.rcParams.update({'font.size': 18})
# Directory containing the JSON files
directory = 'results/throughput_sweep/'

# Range of file indices
file_indices = range(63, 140, 4)

# Initialize a dictionary to hold all the data
all_data = []

# Load each JSON file and extract the data
for idx in file_indices:
    file_name = f'mm_throughput_{idx}.json'
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Extract and sort keys based on the first number
        sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[0]))
        
        # Extract corresponding values
        sorted_values = [data[key] for key in sorted_keys]
        
        # Store the data in the list
        all_data.append(sorted_values)

# Convert list to a numpy array for heatmap
heatmap_data = np.array(all_data)



# Compute the transformed y-axis labels
transformed_y_labels = [idx * 0.352 + 4.10 for idx in file_indices]

# Plotting
plt.figure(figsize=(12, 8))
# plt.imshow(heatmap_data, aspect='auto', cmap='jet', norm=LogNorm(vmin=0.5, vmax=20), origin='lower')
plt.imshow(heatmap_data, aspect='auto', norm=LogNorm(vmin=0.5, vmax=15), origin='lower')

# Set x and y labels
plt.xlabel('Number of 4-bit Experts')
plt.ylabel('Allocated GPU Memory (GB)')
# plt.title('Throughput (token/s)')

# Set x ticks to the number of 4-bit experts
reduced_x_indices = range(0, len(sorted_keys), 2)
plt.xticks(reduced_x_indices, [int(sorted_keys[i].split('_')[0]) for i in reduced_x_indices], rotation=90)


# Set y ticks to the transformed file indices
plt.yticks(range(len(file_indices)), [f'{label:.2f}' for label in transformed_y_labels])

# Add a colorbar to show the scale
plt.colorbar(label='Throughput Value (token/s)')

# Save the heatmap
plt.savefig("plots/throughput_heatmap.png", dpi=400)

# Show the heatmap
plt.show()
