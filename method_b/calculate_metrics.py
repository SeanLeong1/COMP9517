import os
import glob
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import sys

# --- 1. (CRITICAL) Define Paths ---
TRUE_LABEL_DIR = r"D:\UNSW\COMP9517 Project\COMP9517\data\AgroPest-12\test\labels"

PRED_LABEL_DIR = r"D:\UNSW\COMP9517 Project\yolov5\runs\detect\exp2\labels"

THRESHOLD = 0.5

# --- 2. Check if paths exist ---
true_label_files = glob.glob(os.path.join(TRUE_LABEL_DIR, '*.txt'))
if not true_label_files:
    print(f"!!! ERROR: No ground truth label files found in {TRUE_LABEL_DIR}.")
    print("Please check your TRUE_LABEL_DIR path.")
    sys.exit()

print(f"Found {len(true_label_files)} ground truth label files.")

if not os.path.exists(PRED_LABEL_DIR):
    print(f"!!! ERROR: Prediction path not found at {PRED_LABEL_DIR}.")
    print("Please check your PRED_LABEL_DIR path and ensure the 'exp' number is correct.")
    sys.exit()

print(f"Reading predictions from {PRED_LABEL_DIR}...")


# --- 3. Initialize lists for all 12 classes ---
# (From 0 to 11)
for i in range(12):
    locals()[f'y_true_class_{i}'] = []
    locals()[f'y_pred_scores_class_{i}'] = []

# --- 4. Iterate over all "ground truth" label files ---
for true_file_path in true_label_files:
    file_name = os.path.basename(true_file_path)
    
    # A. Get Ground Truth (present classes)
    true_classes_present = set() # Stores which classes are *actually* present in this image
    with open(true_file_path, 'r') as f_true:
        for line in f_true:
            try:
                class_id = int(line.split()[0])
                true_classes_present.add(class_id)
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed ground truth line: {line} (in {file_name})")

    
    # B. Get Prediction Scores
    pred_file_path = os.path.join(PRED_LABEL_DIR, file_name)
    pred_scores_for_this_image = {} # Stores the *highest predicted* score for each class in this image
    
    if os.path.exists(pred_file_path):
        with open(pred_file_path, 'r') as f_pred:
            for line in f_pred:
                try:
                    parts = line.split()
                    class_id = int(parts[0])
                    confidence = float(parts[5]) # YOLO (class x_c y_c w h confidence)
                    
                    # Update the highest score for this class
                    if class_id not in pred_scores_for_this_image or confidence > pred_scores_for_this_image[class_id]:
                        pred_scores_for_this_image[class_id] = confidence
                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed prediction line: {line} (in {file_name})")

    
    # C. Fill the lists for all 12 classes
    for i in range(12):
        # Check if class 'i' is actually present in this image
        is_present = 1 if i in true_classes_present else 0
        locals()[f'y_true_class_{i}'].append(is_present)
        
        # Get the predicted score for class 'i' (defaults to 0.0 if not predicted)
        score = pred_scores_for_this_image.get(i, 0.0)
        locals()[f'y_pred_scores_class_{i}'].append(score)

print("...All images processed. Calculating final metrics...")

# --- 5. Calculate Final Metrics (This is the part you already had) ---
all_true_labels = []
all_pred_labels = []
all_pred_scores = []

for i in range(12):
    y_true = locals()[f'y_true_class_{i}']
    y_scores = locals()[f'y_pred_scores_class_{i}']
    # Convert scores to binary 0 or 1 based on threshold
    y_pred_binary = [1 if score >= THRESHOLD else 0 for score in y_scores]

    all_true_labels.extend(y_true)
    all_pred_labels.extend(y_pred_binary)
    all_pred_scores.extend(y_scores)
    
    # (Optional) Print F1 score for each class
    # class_f1 = f1_score(y_true, y_pred_binary)
    # print(f"Class {i} F1 Score (at {THRESHOLD} threshold): {class_f1:.4f}")

# --- Calculate Overall Metrics ---
try:
    f1_weighted = f1_score(all_true_labels, all_pred_labels, average='weighted')
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    # AUC always uses the raw scores, not the binary labels
    auc_weighted = roc_auc_score(all_true_labels, all_pred_scores, average='weighted')
    
    print("\n--- Final Classification Metrics (Task 3.2) ---")
    print(f"Using Threshold: {THRESHOLD}")
    print(f"Overall Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Weighted AUC: {auc_weighted:.4f}")

except ValueError as e:
    print(f"\n!!! ERROR: Could not calculate metrics.")
    print(f"Error message: {e}")
    print("This usually happens if the 'test/labels' folder is empty or contains no valid labels.")
    print("Please double-check your TRUE_LABEL_DIR path.")