import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib  # Used to save the model
import time
import os

# --- 1. Configuration and Decision Points ---

# (Decision Point) You need to decide on a 'k' value (vocabulary size) [cite from user prompt]
K_CLUSTERS = 500  # Start with k=500 as suggested

# Define the file paths agreed upon by the team
# This is the file Tom (Task 1.1) should output
ALL_DESCRIPTORS_PATH = 'models/sift_all_train_descriptors.npy'

# This is the file you (Task 1.2) will output
VOCABULARY_PATH = 'data/method_b_models/bow_vocabulary.pkl'

# Ensure the output directory exists
os.makedirs(os.path.dirname(VOCABULARY_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ALL_DESCRIPTORS_PATH), exist_ok=True)


# --- 2. Load Data ---

print(f"Loading all SIFT descriptors from {ALL_DESCRIPTORS_PATH}...")
print("This may take some time depending on the file size...")

if not os.path.exists(ALL_DESCRIPTORS_PATH):
    print(f"Error: Descriptor file not found at: {ALL_DESCRIPTORS_PATH}")
    print("Please run Task 1.1 (feature extraction) first.")
    exit()

all_descriptors = np.load(ALL_DESCRIPTORS_PATH)
print(f"Loaded {all_descriptors.shape[0]} SIFT descriptors, each {all_descriptors.shape[1]}-dimensional.")


# --- 3. K-Means Clustering (Your Core Task) ---

print(f"Clustering {all_descriptors.shape[0]} descriptors into {K_CLUSTERS} clusters using MiniBatchKMeans...")

# We use MiniBatchKMeans as it's faster and uses less memory, ideal for large datasets
kmeans = MiniBatchKMeans(
    n_clusters=K_CLUSTERS,
    random_state=42,     # For reproducible results
    batch_size=256,      # Process 256 samples at a time
    n_init='auto',       # Automatically determine number of initializations
    max_iter=100,        # Maximum iterations per batch
    verbose=1            # Print progress
)

start_time = time.time()
kmeans.fit(all_descriptors)
end_time = time.time()

print(f"K-Means clustering finished. Time taken: {end_time - start_time:.2f} seconds.")

# --- 4. Save the Vocabulary ---

# The cluster centers are our "visual vocabulary"
vocabulary = kmeans.cluster_centers_
print(f"Visual vocabulary generated. Shape: {vocabulary.shape}") # Should be (500, 128)

# (Output) Save the vocabulary (the entire KMeans object)
# We save the whole object because it's more useful for encoding later
# Bree (Task 2.1) can load it and use its .predict() method
print(f"Saving the vocabulary (KMeans model) to {VOCABULARY_PATH}...")
joblib.dump(kmeans, VOCABULARY_PATH)

print("--- Task 1.2 (Build Visual Vocabulary) is complete! ---")
print(f"Output file: {VOCABULARY_PATH}")