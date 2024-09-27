import json
import matplotlib.pyplot as plt


def plot_points(points, dataset):

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    plt.clf()
    plt.rcParams.update({'font.size': 14})


    plt.figure(figsize=(8, 6))
    
    plt.scatter(x_values, y_values, zorder=3)
    plt.xlabel('No. 4bit Experts (out of 256)')
    plt.ylabel('Perplexity (lower better)')
    plt.title(f"{dataset}")
    plt.grid(True, zorder=0)
    plt.savefig(f"plots/{dataset}.png", dpi=300)

# Load the data from the JSON file
with open('results/mem_vs_ppl.json', 'r') as file:
    data = json.load(file)

# Initialize lists for each no1 value
wikitext = []
ptb = []
c4 = []


# Iterate through the keys and store values in respective lists
for key in data.keys():
    key_parts = key.rsplit('_', 1)
    dataset = key_parts[0]
    no_4bit_experts = int(key_parts[1])

    if dataset == "wikitext":
        wikitext.append((no_4bit_experts, data[key]))
    elif dataset == "ptb_text_only":
        ptb.append((no_4bit_experts, data[key]))
    elif dataset == "c4":
        c4.append((no_4bit_experts, data[key]))


# Print or do whatever you want with the lists
# print("Values for no1=16:", no1_16)
# print("Values for no1=32:", no1_32)
# print("Values for no1=64:", no1_64)
# print("Values for no1=128:", no1_128)




plot_points(wikitext, "WikiText2")
plot_points(c4, "C4")
plot_points(ptb, "PTB")