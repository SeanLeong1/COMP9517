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
    DATA_ROOT_PATH = os.path.join('..', 'data')
    POS_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'positives')
    NEG_DATA_PATH = os.path.join(DATA_ROOT_PATH, 'negatives')
    MODEL_OUTPUT_PATH = 'svm_model.joblib'
    

    NUM_NEG_SAMPLES = 15282  # Match the number of positive samples for a 1:1 ratio

    # --- 1. Load the Prepared Dataset ---
    print("--- Step 1: Loading Prepared Dataset ---")
    images, labels = load_prepared_dataset(POS_DATA_PATH, NEG_DATA_PATH)
    print(f"Loaded {len(images)} total samples.")

    # --- !! NEW: Subsampling and Balancing Logic !! ---
    print(f"\n--- Subsampling to a balanced dataset ---")
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]

    # Randomly select a subset of negative indices
    random_negative_indices = np.random.choice(negative_indices, size=NUM_NEG_SAMPLES, replace=False)
    
    # Combine the indices
    final_indices = np.concatenate([positive_indices, random_negative_indices])
    np.random.shuffle(final_indices) # Shuffle to mix positives and negatives

    # Create the final balanced dataset
    subset_images = images[final_indices]
    subset_labels = labels[final_indices]
    
    print(f"Using {len(subset_images)} balanced samples for training ({len(positive_indices)} positive, {len(random_negative_indices)} negative).")

    # --- 2. Extract HOG Features (on the smaller subset) ---
    print("\n--- Step 2: Extracting HOG Features ---")
    hog_features = extract_hog_features(subset_images)
    print(f"HOG feature matrix shape: {hog_features.shape}")

    # --- 3. Train SVM Classifier ---
    print("\n--- Step 3: Training SVM Classifier ---")
    svm_model = train_svm(hog_features, subset_labels)

    # --- 4. Save the Trained Model ---
    print("\n--- Step 4: Saving the Trained Model ---")
    save_model(svm_model, MODEL_OUTPUT_PATH)
    
    print("\nTraining pipeline for Method A is complete.")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    main()