import os
import shutil
import json

# Paths to the validation images folder and the JSON file
val_dir = '/raid/sa53869/datasets/imagenet/ILSVRC/Data/CLS-LOC/val'  # Replace with the actual path to your 'val' folder
json_file = '/raid/sa53869/datasets/imagenet/ILSVRC2012_val_labels.json'  # Replace with the actual path to your JSON file

# Create a directory for organized images
organized_dir = '/raid/sa53869/datasets/imagenet/ILSVRC/Data/CLS-LOC/val_2'   # Replace with the desired output folder path

# Ensure the output folder exists
if not os.path.exists(organized_dir):
    os.makedirs(organized_dir)

# Load the JSON file containing image-to-class mapping
with open(json_file, 'r') as f:
    image_class_mapping = json.load(f)

# Organize images into class-named folders
for image_name, class_name in image_class_mapping.items():
    # Source path of the image
    src_path = os.path.join(val_dir, image_name)

    # Create the target class folder if it doesn't exist
    class_folder = os.path.join(organized_dir, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Destination path for the image
    dst_path = os.path.join(class_folder, image_name)

    # Move the image to the class folder
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f'Moved {image_name} to {class_folder}')
    else:
        print(f'Image {image_name} not found in {val_dir}')

print("Organization complete.")
