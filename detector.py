# detector.py
# This script uses the trained SVM model to detect objects in a new image.
# It orchestrates the image pyramid, sliding window, HOG feature extraction,
# classification, and non-maximum suppression steps.

import cv2
import argparse
import numpy as np
from models.svm_classifier import load_model
from features.hog_extractor import extract_hog_features
from detection.sliding_window import pyramid, sliding_window
from detection.nms import non_maximum_suppression

def detector():
    """
    Main function to run the object detection pipeline.
    """
    # --- 1. Argument Parsing ---
    # Set up a command-line interface to make the script easy to use.
    ap = argparse.ArgumentParser(description="HOG + Linear SVM Object Detector")
    ap.add_argument("-i", "--image", required=True, help="Path to the image for detection")
    ap.add_argument("-m", "--model", required=True, help="Path to the trained SVM model (.joblib)")
    ap.add_argument("-c", "--confidence", type=float, default=1.2, help="Confidence threshold for a detection")
    args = vars(ap.parse_args())

    # --- 2. Configuration & Model Loading ---
    # These parameters must match the ones used during training.
    WIN_WIDTH = 64
    WIN_HEIGHT = 128
    WINDOW_SIZE = (WIN_WIDTH, WIN_HEIGHT)
    
    # These can be tuned for detection performance vs. speed.
    PYRAMID_SCALE = 1.5
    STEP_SIZE = 16 # How many pixels to skip in x and y
    NMS_THRESHOLD = 0.5 # How much overlap is allowed

    print("--- Loading SVM Model ---")
    model = load_model(args["model"])

    # --- 3. Image Loading & Pyramid Generation ---
    image = cv2.imread(args["image"])
    if image is None:
        print(f"Error: Could not load image at {args['image']}")
        return
        
    # We keep the original image for drawing final boxes.
    original_image = image.copy()
    
    # The list to store all potential detections before NMS.
    detections = []
    current_scale = 1.0

    print("--- Scanning Image with Pyramid and Sliding Window ---")
    # Loop over the image pyramid to detect objects at different scales.
    for scaled_image in pyramid(image, scale=PYRAMID_SCALE, min_size=(WIN_WIDTH, WIN_HEIGHT)):
        # The sliding window operates on each layer of the pyramid.
        for (x, y, window) in sliding_window(scaled_image, step_size=STEP_SIZE, window_size=WINDOW_SIZE):
            
            # Ensure the window is the correct size (it might not be at the image edges).
            if window.shape[0] != WIN_HEIGHT or window.shape[1] != WIN_WIDTH:
                continue

            # --- 4. Feature Extraction and Classification ---
            # Extract HOG features from the current window.
            # The function expects a list of images, so we wrap `window`.
            window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            features = extract_hog_features([window_gray])
            
            # Use the decision_function to get a confidence score. A positive score
            # indicates a potential object.
            confidence_score = model.decision_function(features)

            # If the score is high enough, we record this as a potential detection.
            if confidence_score > args["confidence"]:
                # The coordinates (x, y) are relative to the *scaled* image.
                # We must scale them back to the original image's coordinate system.
                scaled_x = int(x * current_scale)
                scaled_y = int(y * current_scale)
                scaled_w = int(WIN_WIDTH * current_scale)
                scaled_h = int(WIN_HEIGHT * current_scale)
                
                # Append (startX, startY, endX, endY, confidence)
                box = (scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h, confidence_score[0])
                detections.append(box)

        # Update the scale for the next pyramid layer.
        current_scale *= PYRAMID_SCALE
        
    print(f"Found {len(detections)} potential detections before NMS.")

    # --- 5. Non-Maximum Suppression ---
    print("--- Applying Non-Maximum Suppression ---")
    # Convert detections to a NumPy array for NMS processing.
    detections = np.array(detections)
    
    # Apply NMS to merge overlapping boxes into single, confident detections.
    final_boxes = non_maximum_suppression(detections, NMS_THRESHOLD)
    print(f"Found {len(final_boxes)} final detections after NMS.")

    # --- 6. Visualization ---
    # Loop over the final detections and draw them on the original image.
    for (startX, startY, endX, endY, score) in final_boxes:
        cv2.rectangle(
            original_image,
            (int(startX), int(startY)),
            (int(endX), int(endY)),
            (0, 255, 0), # Green box
            2
        )
        label = f"Insect: {score:.2f}"
        cv2.putText(
            original_image,
            label,
            (int(startX), int(startY) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display the final results.
    cv2.imshow("Detections", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector()