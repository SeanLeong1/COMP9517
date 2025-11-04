# models/svm_classifier.py
# This module handles the training, saving, and loading of the SVM classifier.

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_svm(features, labels):
    """
    Trains a Support Vector Machine (SVM) classifier.

    This function takes feature vectors and their corresponding labels, splits them
    into training and testing sets, trains an SVM model, and evaluates its performance.

    Args:
        features (np.ndarray): A 2D array of feature vectors.
        labels (np.ndarray): A 1D array of corresponding labels.

    Returns:
        SVC: The trained Scikit-learn SVM model object.
    """
    # Split the data into training and validation sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Training SVM classifier...")
    # Initialize the SVM. A linear kernel is fast and a good baseline.
    # The 'C' parameter is a crucial hyperparameter for regularization.
    model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the held-out test set
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model

def save_model(model, filepath):
    """
    Saves the trained model to a file.

    Args:
        model (SVC): The trained Scikit-learn model object.
        filepath (str): The path where the model will be saved.
    """
    print(f"Saving model to {filepath}")
    # joblib is efficient for saving scikit-learn models
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Loads a trained model from a file.

    Args:
        filepath (str): The path from which to load the model.

    Returns:
        SVC: The loaded Scikit-learn model object.
    """
    print(f"Loading model from {filepath}")
    return joblib.load(filepath)