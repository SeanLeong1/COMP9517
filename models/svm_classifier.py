# models/svm_classifier.py
# This module handles the training, saving, and loading of the SVM classifier.

# IMPORTANT: We are now using LinearSVC for performance with large datasets.
from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_svm(features, labels):
    """
    Trains a Linear Support Vector Machine (LinearSVC) classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Training LinearSVC classifier...")
    # LinearSVC is much faster for linear kernels and large sample sizes.
    # It is a good practice to increase `max_iter` for convergence.
    model = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model

# The save_model and load_model functions remain the same.
def save_model(model, filepath):
    print(f"Saving model to {filepath}")
    joblib.dump(model, filepath)

def load_model(filepath):
    print(f"Loading model from {filepath}")
    return joblib.load(filepath)