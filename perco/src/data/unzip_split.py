import os
import zipfile

def unzip_files(source_folder, destination_folder):
    # Ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all the files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a zip file
        if filename.endswith('.zip'):
            zip_file_path = os.path.join(source_folder, filename)
            # Create a ZipFile object
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Extract all contents into the destination folder
                zip_ref.extractall(destination_folder)

# Usage example
unzip_files(source_folder='/scratch/09004/sravana/OpenImagesV6_split', destination_folder='/scratch/09004/sravana/OpenImagesV6_split/train')