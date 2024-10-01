import os
import shutil

relevant_name = '/work/09004/sravana/ls6/ncomp/perco/res/doc/coco_names.txt'
mscoco2017_train = '/scratch/09004/sravana/MSCOCO/train2017'
filtered_mscoco2017_train = '/scratch/09004/sravana/MSCOCO/MSCOCO30k'

# Read relevant names from file
with open(relevant_name, 'r') as file:
    relevant_names = set(file.read().splitlines())
    
# Ensure the output directory exists
os.makedirs(filtered_mscoco2017_train, exist_ok=True)


# Iterate over images in mscoco2017_train directory
for filename in os.listdir(mscoco2017_train):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(mscoco2017_train, filename)
        # Check if image name is in the relevant names set
        if filename in relevant_names:
            # Copy the image to the filtered_mscoco2017_train directory
            shutil.copy(image_path, filtered_mscoco2017_train)