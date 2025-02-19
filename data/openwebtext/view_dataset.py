import os
from datasets import load_from_disk

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the dataset
dataset_path = os.path.join(current_directory, "dataset_cache")

# Load the dataset from the constructed path
dataset = load_from_disk(dataset_path)

# View the first 5 samples in the train split
print("First 5 samples:")
for i in range(5):
    print(dataset['train'][i]['text'])

# Define the number of random samples you want to view
import random
num_samples = 5

# Get random indices
random_indices = random.sample(range(len(dataset['train'])), num_samples)

# Print the random samples
print("\nRandom samples:")
for idx in random_indices:
    print(dataset['train'][idx]['text'])

# Shuffle the dataset and select the first few samples
shuffled_dataset = dataset['train'].shuffle(seed=42)

# View the first 5 samples in the shuffled dataset
print("\nShuffled samples:")
for i in range(5):
    print(shuffled_dataset[i]['text'])
