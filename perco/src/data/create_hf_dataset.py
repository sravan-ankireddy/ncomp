import os
import random
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

# Function to read image as raw bytes
def read_image_as_bytes(image_path):
    """Reads an image and returns its raw byte content."""
    with open(image_path, "rb") as f:
        return f.read()

# Load captions from results.txt into a dictionary
def load_captions(captions_txt_path):
    """Loads the base image ID and captions from the results.txt CSV file."""
    captions = {}
    with open(captions_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 2:  # Ensure there's at least base_image_id and caption
                base_image_id = parts[0]
                caption = parts[1]  # The second column is the caption
                captions[base_image_id] = caption
    return captions

# Create and save dataset in chunks
def create_and_save_dataset_in_chunks(image_dirs, captions_dirs, output_dir, num_samples=10000, chunk_size=10000):
    # Load captions for each compression level
    captions_dicts = {rate: load_captions(os.path.join(rate, "results.txt")) for rate in captions_dirs}

    # Get all image filenames from the first rate folder (assuming filenames are consistent across folders)
    filenames = os.listdir(os.path.join(image_dirs[0], "patches_fake"))

    sampled_pairs = set()  # To track unique (image, rate_0, rate_1) pairs

    chunk_id = 1
    for i in tqdm(range(0, num_samples, chunk_size), desc="Processing in chunks"):
        data = {"jpg_0": [], "jpg_1": [], "label_0": [], "caption": []}

        # Sample a chunk of random images
        while len(data["jpg_0"]) < chunk_size:
            # Randomly select an image
            random_image_filename = random.choice(filenames)
            base_image_id = random_image_filename.split('_')[0]

            # Randomly select two different rates
            rate_0, rate_1 = random.sample(image_dirs, 2)

            # Check if this pair has been sampled before (order doesn't matter)
            pair = tuple(sorted([random_image_filename, rate_0, rate_1]))
            if pair in sampled_pairs:
                continue  # Skip if we've already sampled this combination

            # Mark this pair as sampled
            sampled_pairs.add(pair)

            # Create the image paths for the selected rates
            image_0_path = os.path.join(rate_0, "patches_fake", random_image_filename)
            image_1_path = os.path.join(rate_1, "patches_fake", random_image_filename)

            # Read the images as bytes
            img_bytes_0 = read_image_as_bytes(image_0_path)
            img_bytes_1 = read_image_as_bytes(image_1_path)

            # Label is 0 if rate_0 is less compressed than rate_1, else 1
            rate_val_0 = float(os.path.basename(rate_0))
            rate_val_1 = float(os.path.basename(rate_1))
            label_0 = 0 if rate_val_0 < rate_val_1 else 1

            # Retrieve the caption from the corresponding results.txt
            if base_image_id in captions_dicts[rate_0]:
                caption = captions_dicts[rate_0][base_image_id]
            elif base_image_id in captions_dicts[rate_1]:  # Fallback to the other rate if not in the first
                caption = captions_dicts[rate_1][base_image_id]
            else:
                raise KeyError(f"Caption not found for {base_image_id}")

            # Store in dataset
            data["jpg_0"].append(img_bytes_0)
            data["jpg_1"].append(img_bytes_1)
            data["label_0"].append(label_0)
            data["caption"].append(caption)

        # Convert the chunk to a DataFrame and save it as Parquet
        df = pd.DataFrame(data)
        part_path = os.path.join(output_dir, f'dataset_part_{chunk_id}.parquet')
        df.to_parquet(part_path, index=False)
        print(f"Saved {part_path}")

        chunk_id += 1

# Usage example
image_dirs = ['/raid/sa53869/datasets/OpenImagesV6_sample_v0/synth/0.0019', 
              '/raid/sa53869/datasets/OpenImagesV6_sample_v0/synth/0.0313', 
              '/raid/sa53869/datasets/OpenImagesV6_sample_v0/synth/0.1250']  # Directories for different compression rates
captions_dirs = image_dirs  # Assuming the results.txt is inside these directories
output_dir = '/raid/sa53869/datasets/OpenImagesV6_sample_v0/synth/parquet'

create_and_save_dataset_in_chunks(image_dirs, captions_dirs, output_dir, num_samples=15000, chunk_size=15000)

# Load the dataset to verify
parquet_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('dataset_part')]
dataset = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
print(f"Loaded dataset columns: {dataset.columns}")
print(f"Number of samples: {len(dataset)}")

# # Optionally, you can iterate over and verify image data by converting raw bytes back to images
# for _, item in dataset.iterrows():
#     img_0 = Image.open(io.BytesIO(item["jpg_0"])).convert("RGB")
#     img_1 = Image.open(io.BytesIO(item["jpg_1"])).convert("RGB")

# print("Dataset successfully created and verified.")
