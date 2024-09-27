import json
import matplotlib.pyplot as plt
import os


plt.rcParams.update({'font.size': 18})
# Directory containing the JSON files
directory = 'results/throughput_sweep/'

# Range of file indices
file_indices = range(67,140,8)

transformed_y_labels = [idx * 0.352 + 4.10 for idx in file_indices]

# Initialize a dictionary to hold all the data
all_data = {}

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
        
        # Extract x-axis labels
        x_labels = [int(key.split('_')[0]) for key in sorted_keys]
        
        # Store the data in the dictionary
        all_data[idx] = (x_labels, sorted_values)



# print(all_data)
# Plotting
plt.figure(figsize=(10, 8))



for idx, (x_labels, sorted_values) in all_data.items():
    plt.plot(x_labels, sorted_values, label=f'{idx * 0.352 + 4.10:.2f} GB', linewidth = 2)

plt.xlabel('Number of 4-bit Experts')
plt.ylabel('Throughput (token/s)')
plt.title('')
plt.legend()
plt.grid(True)
plt.savefig("plots/throughput_sweep.png", dpi=400)
