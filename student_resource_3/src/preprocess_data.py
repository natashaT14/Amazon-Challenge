import cv2
import numpy as np
from PIL import Image, ImageFile
import os

# To handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Resize Image
def resize_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size, Image.LANCZOS)  # Use Image.LANCZOS for high-quality resizing
    return img_resized

# Normalize Image
def normalize_image(image):
    image_np = np.array(image) / 255.0
    return Image.fromarray((image_np * 255).astype(np.uint8))

# Data Augmentation
def augment_image(image):
    # Convert to numpy array for augmentation
    image_np = np.array(image)
    
    # Horizontal Flip
    if np.random.rand() > 0.5:
        image_np = np.fliplr(image_np)
    
    # Rotation
    angle = np.random.randint(-30, 30)  # Rotate between -30 to 30 degrees
    image_pil = Image.fromarray(image_np)
    image_np = np.array(image_pil.rotate(angle))
    
    return Image.fromarray(image_np)

# Process All Images
def process_images(input_folder, output_folder, target_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            try:
                # Resize image
                image_resized = resize_image(image_path, target_size)
                
                # Normalize image
                image_normalized = normalize_image(image_resized)
                
                # Augment image
                image_augmented = augment_image(image_normalized)
                
                # Save processed image
                output_path = os.path.join(output_folder, filename)
                image_augmented.save(output_path)
                
                print(f"Processed and saved: {filename}")
            except OSError as e:
                print(f"Error processing {filename}: {e}")

# Specify folders
input_folder = 'downloaded_images/'
output_folder = 'processed_images/'

# Process images
process_images(input_folder, output_folder)
