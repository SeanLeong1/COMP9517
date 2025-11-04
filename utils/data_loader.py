# utils/data_loader.py
# This module is responsible for loading and preprocessing the AgroPest-12 dataset.

import os
import cv2

def load_dataset(dataset_path):
    """
    Loads images and their corresponding labels from the dataset directory.

    Args:
        dataset_path (str): The path to the root of the dataset.

    Returns:
        tuple: A tuple containing two lists:
               - A list of images (as NumPy arrays).
               - A list of corresponding labels (as strings or integers).
    """
    images = []
    labels = []
    
    # Assume the dataset is structured with subdirectories for each class
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                
                # Read the image in grayscale for HOG, or color if needed
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    images.append(image)
                    labels.append(class_folder)
                    
    return images, labels