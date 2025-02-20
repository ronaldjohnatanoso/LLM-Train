import os
from datasets import load_dataset

# Set the cache directory to the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_DATASETS_CACHE"] = os.path.join(current_directory, "dataset_cache")

# Number of workers in load_dataset() call
num_proc_load_dataset = 8

if __name__ == '__main__':
    # Create the cache directory if it doesn't exist
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

    # Download and cache the dataset
    print("Downloading the dataset...")
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)
    
    # Save the dataset locally
    dataset_dir = os.path.join(current_directory, "dataset_cache")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset.save_to_disk(dataset_dir)
    print(f"Dataset downloaded and saved to {dataset_dir}")
