# utils/data_loader.py
# This module is responsible for loading the prepared positive and negative
# training samples for the detector.

import os
import cv2
import numpy as np
from tqdm import tqdm

def load_prepared_dataset(positive_path, negative_path):
    """
    Loads pre-processed positive and negative image samples.

    Args:
        positive_path (str): Path to the directory of positive samples.
        negative_path (str): Path to the directory of negative samples.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - A NumPy array of images.
               - A NumPy array of corresponding labels (1 for positive, 0 for negative).
    """
    images = []
    labels = []

    # Load positive samples and assign label 1
    print("Loading positive samples...")
    for filename in tqdm(os.listdir(positive_path)):
        if filename.endswith('.jpg'):
            image = cv2.imread(os.path.join(positive_path, filename), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(1)

    # Load negative samples and assign label 0
    print("Loading negative samples...")
    for filename in tqdm(os.listdir(negative_path)):
        if filename.endswith('.jpg'):
            image = cv2.imread(os.path.join(negative_path, filename), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(0)

    return np.array(images), np.array(labels)