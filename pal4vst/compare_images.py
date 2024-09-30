import os
from PIL import Image, ImageDraw, ImageFont

def stitch_images(reference_folder_path, folder_path, subfolder1, subfolder2, output_pdf_path):
    # Define paths for the two subfolders and the combined folder
    subfolder1_path = os.path.join(folder_path, subfolder1)
    subfolder2_path = os.path.join(folder_path, subfolder2)
    combined_folder_path = os.path.join(folder_path, 'combined')

    # Create the 'combined' folder if it doesn't exist
    os.makedirs(combined_folder_path, exist_ok=True)

    # Set the labels
    labels = ["original", "reconstructed", "heat map"]

    # Get list of images from the reference folder
    image_names = os.listdir(reference_folder_path)

    # Load a default font (optional: you can specify a different font if you want)
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Try to use a specific font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # List to store all combined images for the PDF
    combined_images = []

    # Process each image
    for img_name in image_names:
        ref_img_path = os.path.join(reference_folder_path, img_name)
        img1_path = os.path.join(subfolder1_path, img_name)
        img2_path = os.path.join(subfolder2_path, img_name)

        # Ensure the image exists in both subfolders
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Image {img_name} is missing in {subfolder1_path} or {subfolder2_path}")
            continue

        # Open images from the reference folder and both subfolders
        ref_img = Image.open(ref_img_path)
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Ensure all images have the same height
        if ref_img.height != img1.height or ref_img.height != img2.height:
            print(f"Image {img_name} heights do not match. Resizing images to match reference image height.")
            img1 = img1.resize((img1.width, ref_img.height))
            img2 = img2.resize((img2.width, ref_img.height))

        # Define the space for the label (increase the image height)
        label_height = 30  # Height for the labels at the bottom

        # Create a new image with combined width and extra height for the labels
        combined_width = ref_img.width + img1.width + img2.width
        combined_height = ref_img.height + label_height
        combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

        # Paste the three images side by side
        combined_image.paste(ref_img, (0, 0))  # ref_img as the first column
        combined_image.paste(img1, (ref_img.width, 0))  # img1 in the middle
        combined_image.paste(img2, (ref_img.width + img1.width, 0))  # img2 as the last column

        # Draw the labels under each image
        draw = ImageDraw.Draw(combined_image)

        # Calculate positions for labels using textbbox
        label_positions = []
        for i, label in enumerate(labels):
            width_offset = ref_img.width * i if i == 0 else ref_img.width + img1.width * (i - 1)
            bbox = draw.textbbox((0, 0), label, font=font)  # Get bounding box for text
            text_width = bbox[2] - bbox[0]  # Calculate text width
            label_positions.append((width_offset + (img1.width - text_width) // 2, ref_img.height))

        # Draw labels
        for i, label in enumerate(labels):
            draw.text(label_positions[i], label, font=font, fill="black")

        # Save the combined image
        combined_image.save(os.path.join(combined_folder_path, img_name))

        # Append the combined image to the list for PDF creation
        combined_images.append(combined_image.convert('RGB'))  # Ensure all images are RGB

        print(f"Combined image with labels saved: {img_name}")

    # Save all combined images as a single PDF
    if combined_images:
        combined_images[0].save(output_pdf_path, save_all=True, append_images=combined_images[1:])
        print(f"PDF saved at {output_pdf_path}")

# Example usage
bpp = 0.0548
reference_folder_path = "/home/sa53869/pal4vst/taco_data/Kodak"  # Replace with your reference folder path
folder_path = f"/home/sa53869/pal4vst/taco_data/bpp_{bpp}"  # Replace with your folder path
subfolder1 = "images"  # Replace with your first subfolder name
subfolder2 = "vis_pal"  # Replace with your second subfolder name
output_pdf_path = f"/home/sa53869/pal4vst/taco_data/bpp_{bpp}/combined_images_{bpp}.pdf"  # Replace with your output PDF path

stitch_images(reference_folder_path, folder_path, subfolder1, subfolder2, output_pdf_path)
