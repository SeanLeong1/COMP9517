# main.py
# Main script to execute the computer vision training pipeline.

import os
import numpy as np
from utils.data_loader import load_prepared_dataset
from features.hog_extractor import extract_hog_features
from models.svm_classifier import train_svm, save_model

def main():
    """
    Main function to run the feature extraction and classification pipeline.
    """
    # --- Configuration ---
    # Define paths for our prepared data.
    POS_DATA_PATH = 'data/positives'
    NEG_DATA_PATH = 'data/negatives'
    MODEL_OUTPUT_PATH = 'svm_model.joblib'

    # --- 1. Load the Prepared Dataset ---
    print("--- Step 1: Loading Prepared Dataset ---")
    images, labels = load_prepared_dataset(POS_DATA_PATH, NEG_DATA_PATH)
    print(f"Loaded {len(images)} total samples ({np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative).")

    # --- 2. Extract HOG Features (Method A) ---
    print("\n--- Step 2: Extracting HOG Features ---")
    # Note: The target_size in extract_hog_features must match the size
    # used in prepare_data.py. Default is (64, 128).
    hog_features = extract_hog_features(images)
    print(f"HOG feature matrix shape: {hog_features.shape}")

    # --- 3. Train SVM Classifier ---
    print("\n--- Step 3: Training SVM Classifier ---")
    svm_model = train_svm(hog_features, labels)

    # --- 4. Save the Trained Model ---
    print("\n--- Step 4: Saving the Trained Model ---")
    save_model(svm_model, MODEL_OUTPUT_PATH)
    
    print("\nTraining pipeline for Method A is complete.")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    main()