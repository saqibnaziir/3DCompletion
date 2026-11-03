import cv2
import numpy as np
import os
from pathlib import Path

def find_content_bounds_by_pixel_change(image):
    """
    Find image bounds by detecting actual pixel value changes
    """
    if len(image.shape) == 3:
        # Convert to grayscale while handling alpha channel
        if image.shape[2] == 4:
            # Use alpha channel as part of detection
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Combine with alpha
            gray = cv2.addWeighted(gray, 0.5, alpha, 0.5, 0)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate pixel differences
    diff_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
    diff_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
    
    # Find where significant changes occur
    threshold = 1.0  # Threshold for considering a change significant
    
    # Find bounds in x direction (columns)
    col_changes = np.where(diff_x.max(axis=0) > threshold)[0]
    if len(col_changes) == 0:
        return None
    x_start = col_changes[0]
    x_end = col_changes[-1] + 2  # +2 because diff reduces size by 1
    
    # Find bounds in y direction (rows)
    row_changes = np.where(diff_y.max(axis=1) > threshold)[0]
    if len(row_changes) == 0:
        return None
    y_start = row_changes[0]
    y_end = row_changes[-1] + 2  # +2 because diff reduces size by 1
    
    # Calculate width and height
    w = x_end - x_start
    h = y_end - y_start
    
    # Add minimal padding of 1 pixel to avoid cutting off edges
    x_start = max(0, x_start - 1)
    y_start = max(0, y_start - 1)
    w = min(image.shape[1] - x_start, w + 2)
    h = min(image.shape[0] - y_start, h + 2)
    
    return x_start, y_start, w, h

def get_max_dimensions(input_folder):
    """
    Find the maximum required dimensions across all cropped images
    """
    max_width = 0
    max_height = 0
    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                continue
                
            bounds = find_content_bounds_by_pixel_change(image)
            if bounds is None:
                continue
                
            _, _, w, h = bounds
            max_width = max(max_width, w)
            max_height = max(max_height, h)
    
    return max_width, max_height

def pad_to_size(image, target_width, target_height):
    """
    Pad image to target size while keeping content centered
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        if c == 4:
            # For images with alpha channel
            padded = np.zeros((target_height, target_width, 4), dtype=image.dtype)
            padded[:, :, 3] = 0  # Set alpha to transparent
        else:
            # For RGB images
            padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    else:
        # For grayscale images
        h, w = image.shape
        padded = np.zeros((target_height, target_width), dtype=image.dtype)
    
    # Calculate padding
    pad_y = (target_height - h) // 2
    pad_x = (target_width - w) // 2
    
    # Copy the image to the center of the padded array
    if len(image.shape) == 3:
        padded[pad_y:pad_y+h, pad_x:pad_x+w, :] = image
    else:
        padded[pad_y:pad_y+h, pad_x:pad_x+w] = image
    
    return padded

def process_images(input_folder, output_folder):
    """
    Process images using pixel change detection for precise cropping
    and ensure uniform output size
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get maximum dimensions needed
    max_width, max_height = get_max_dimensions(input_folder)
    print(f"Maximum dimensions required: {max_width}x{max_height}")
    
    # Supported image extensions
    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(extensions):
            # Read image with alpha channel if present
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Could not read image: {filename}")
                continue
            
            # Find content bounds based on pixel changes
            bounds = find_content_bounds_by_pixel_change(image)
            if bounds is None:
                print(f"Could not find content bounds in image: {filename}")
                continue
            
            x, y, w, h = bounds
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            
            # Pad to uniform size
            final_image = pad_to_size(cropped, max_width, max_height)
            
            # Save the processed image
            output_path = os.path.join(output_folder, f'cropped_{filename}')
            
            # Save with alpha channel if present
            if len(image.shape) == 3 and image.shape[2] == 4:
                cv2.imwrite(output_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(output_path, final_image)
            
            print(f"Processed {filename} - Original: {image.shape}, Cropped: {cropped.shape}, Final: {final_image.shape}")

if __name__ == "__main__":
    # Example usage
    input_folder = r"D:\COURSAI\3DGeoMeshNet\Complete\oliver\7th\COMA\Final\5_Post\Error_vis"
    output_folder =r"D:\COURSAI\3DGeoMeshNet\Complete\oliver\7th\COMA\Final\5_Post\Error_vis/cropped"
    
    # Process the images with uniform size output
    process_images(input_folder, output_folder)
