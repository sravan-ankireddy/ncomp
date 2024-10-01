import tensorflow_datasets as tfds

# Specify the dataset name (e.g., "open_images_v6") and the directory
dataset_name = "open_images_v6"
data_dir = "../data/OpenImagesV6"

# Download the dataset and store it in the specified directory
dataset, info = tfds.load(dataset_name, data_dir=data_dir, with_info=True)

print(f"Dataset downloaded to: {data_dir}")
