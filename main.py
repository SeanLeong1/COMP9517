# main.py
# Main script to execute the computer vision training pipeline.

import os
import numpy as np
from utils.data_loader import load_prepared_dataset
from features.hog_extractor import extract_hog_features
from models.svm_classifier import train_svm, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

    # --- 3. Train & Evaluate SVM Classifier ---
    print("\n--- Step 3: Training & Evaluating SVM Classifier ---")

    # Split into train/validation for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        hog_features,
        subset_labels,
        test_size=0.2,          # 20% for validation
        random_state=42,        # for reproducibility
        stratify=subset_labels  # keep class balance
    )

    # Train SVM on the training split
    svm_model = train_svm(X_train, y_train)

    # Predict on the validation split
    y_pred = svm_model.predict(X_val)

    # --- Extra metrics: classification report + confusion matrix ---
    print("\nClassification report (Method A - HOG + SVM):")
    report = classification_report(y_val, y_pred, digits=4)
    print(report)

    # Save the classification report to a text file (for the report/slides)
    with open("methodA_report.txt", "w") as f:
        f.write(report)

    # Compute and print confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix:\n", cm)

    # Save confusion matrix as CSV so you can make a table/heatmap later
    np.savetxt("methodA_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # --- 4. Save the Trained Model ---
    print("\n--- Step 4: Saving the Trained Model ---")
    save_model(svm_model, MODEL_OUTPUT_PATH)
    
    print("\nTraining pipeline for Method A is complete.")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    main()