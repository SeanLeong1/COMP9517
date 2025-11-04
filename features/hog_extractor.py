# features/hog_extractor.py
# This module computes HOG features for a list of images.

import cv2
import numpy as np
from tqdm import tqdm # For a progress bar

def extract_hog_features(images, target_size=(64, 128)):
    """
    Extracts HOG features from a list of images.

    Args:
        images (list): A list of images (NumPy arrays).
        target_size (tuple): The size (width, height) to resize images to.

    Returns:
        np.ndarray: A 2D array where each row is a HOG feature vector.
    """
    hog = cv2.HOGDescriptor()
    hog_features = []

    print("Extracting HOG features...")
    for image in tqdm(images):
        # HOG requires a fixed image size.
        resized_image = cv2.resize(image, target_size)

        # Compute the HOG features
        features = hog.compute(resized_image)

        # Flatten the features to a 1D vector and append
        hog_features.append(features.flatten())

    return np.array(hog_features)
