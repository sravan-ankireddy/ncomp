import os
import random
import shutil

def copy_random_images(source_folder, destination_folder, num_images=20000):
	# Ensure the destination folder exists
	os.makedirs(destination_folder, exist_ok=True)

	# Get all image files from the source folder (jpg, png, etc.)
	image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
	image_files = [f for f in os.listdir(source_folder) 
				   if f.lower().endswith(image_extensions)]

	if len(image_files) < num_images:
		print(f"Not enough images. Only found {len(image_files)} images.")
		return

	# Randomly select images
	selected_images = random.sample(image_files, num_images)

	# Copy selected images to the destination folder
	for image in selected_images:
		src_path = os.path.join(source_folder, image)
		dst_path = os.path.join(destination_folder, image)
		shutil.copy(src_path, dst_path)

	print(f"Copied {num_images} images to {destination_folder}")

# Example usage
source_folder = "/raid/sa53869/datasets/OpenImagesV6/train"
destination_folder = "/raid/sa53869/datasets/OpenImagesV6_sample/train"
copy_random_images(source_folder, destination_folder)
