# main.py
# Main script to execute the computer vision pipeline.

from utils.data_loader import load_dataset
from features.hog_extractor import extract_hog_features

def main():
    """
    Main function to run the feature extraction and classification pipeline.
    """
    # Define the path to your training data
    dataset_path = 'data/train' # Adjust this path as needed

    # 1. Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    images, labels = load_dataset(dataset_path)
    print(f"Loaded {len(images)} images.")

    # 2. Extract HOG features (Method A)
    hog_features = extract_hog_features(images)
    print(f"HOG feature matrix shape: {hog_features.shape}")

    # 3. Train SVM classifier using hog_features and labels
    # 4. Evaluate the classifier

if __name__ == '__main__':
    main()