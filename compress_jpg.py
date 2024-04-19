import os
from PIL import Image

def compress_images_in_folder(source_folder, output_folder, start, end, quality=85):
    """
    Compress images in the specified folder to JPEG format with the specified quality, 
    choosing images of start < index < end in an increasing ordering of image filenames.
    
    Args:
    source_folder (str): The path to the directory containing images to compress.
    output_folder (str): The path to the directory where compressed images will be saved.
    quality (int): The compression quality for the JPEG files (1-100).

    """
    
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    filenames = sorted(os.listdir(source_folder))[start:end]
    total_input_size, total_output_size = 0, 0

    # Process each file in the source folder
    for filename in filenames:
        file_path = os.path.join(source_folder, filename)
        total_input_size += os.path.getsize(file_path) / 1000
        
        # Attempt to open the file as an image
        try:
            with Image.open(file_path) as img:
                # Define the output file path
                output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
                
                # Convert image to RGB to ensure compatibility with JPEG format
                rgb_img = img.convert('RGB')
                
                # Save the image with the specified compression quality
                rgb_img.save(output_file_path, 'JPEG', quality=quality)
                total_output_size += os.path.getsize(output_file_path) / 1000

        except Exception as e:
            print(f"Failed to compress {filename}: {e}")
            
    return total_input_size, total_output_size

# Example usage
source_dir = 'celeba/img_align_celeba'  # Source folder containing the images
output_dir = 'compressed_jpg'  # Output folder for compressed images
compression_quality = 1  # Compression quality (1-100, where 100 is least compression)

total_input_size, total_output_size = compress_images_in_folder(source_dir, output_dir, 20000, 30000, compression_quality)

print(f"Total compressed from {total_input_size}K to {total_output_size}K. "
      f"Compression Ratio={total_input_size / total_output_size}")