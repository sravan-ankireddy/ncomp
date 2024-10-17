import os

def list_all_file_names(image_folder, output_txt):
    file_count = 0
    
    # Open the output file to write file names incrementally
    with open(output_txt, 'w') as f:
        for root, dirs, files in os.walk(image_folder):
            for file_name in files:
                f.write(file_name + '\n')
                file_count += 1
    
    # Print the total number of files found
    print(f"Total number of files found: {file_count}")

# Example usage:
image_folder = '/scratch/09004/sravana/OpenImagesV6_split/train'
output_txt = 'list_train_files.txt'
list_all_file_names(image_folder, output_txt)


