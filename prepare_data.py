# prepare_data.py
# This script processes the raw AgroPest-12 dataset to generate positive
# and negative image samples for training an object detector.

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_annotations(annotation_path):
    """
    Parses an XML annotation file to extract bounding box coordinates.
    
    Args:
        annotation_path (str): Path to the XML annotation file.
        
    Returns:
        list: A list of bounding box dictionaries, e.g.,
              [{'label': 'insect', 'xmin': 10, 'ymin': 20, 'xmax': 50, 'ymax': 80}, ...]
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    for member in root.findall('object'):
        box = {
            'label': member.find('name').text,
            'xmin': int(member.find('bndbox/xmin').text),
            'ymin': int(member.find('bndbox/ymin').text),
            'xmax': int(member.find('bndbox/xmax').text),
            'ymax': int(member.find('bndbox/ymax').text),
        }
        boxes.append(box)
    return boxes

def generate_positive_samples(image_path, annotation_path, output_dir, target_size=(64, 128)):
    """
    Crops positive samples from an image using its annotations.
    """
    image = cv2.imread(image_path)
    boxes = parse_annotations(annotation_path)
    
    for i, box in enumerate(boxes):
        # Crop the bounding box region from the image
        cropped = image[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        
        # Resize to the fixed size required by our classifier
        resized = cv2.resize(cropped, target_size)
        
        # Save the positive sample
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_pos_{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), resized)

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def generate_negative_samples(image_path, annotation_path, output_dir, target_size=(64, 128), num_samples=10):
    """
    Generates negative samples by randomly sampling patches that do not
    overlap with any ground-truth bounding boxes.
    """
    image = cv2.imread(image_path)
    gt_boxes = parse_annotations(annotation_path)
    
    height, width, _ = image.shape
    samples_generated = 0
    
    # Try to generate num_samples negative samples
    for _ in range(num_samples * 5): # Try more times than needed
        if samples_generated >= num_samples:
            break
            
        # Generate a random window
        x = np.random.randint(0, width - target_size[0])
        y = np.random.randint(0, height - target_size[1])
        random_window = [x, y, x + target_size[0], y + target_size[1]]
        
        is_negative = True
        # Check for overlap with all ground-truth boxes
        for box in gt_boxes:
            gt_box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            if calculate_iou(random_window, gt_box) > 0.1: # Allow for tiny overlap
                is_negative = False
                break
        
        if is_negative:
            # Crop, resize, and save the negative sample
            cropped = image[y:y + target_size[1], x:x + target_size[0]]
            resized = cv2.resize(cropped, target_size)
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_neg_{samples_generated}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), resized)
            samples_generated += 1

def process_dataset(image_dir, annotation_dir, pos_output_dir, neg_output_dir):
    """
    Main function to process the entire dataset.
    """
    # Create output directories if they don't exist
    os.makedirs(pos_output_dir, exist_ok=True)
    os.makedirs(neg_output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        # Assume annotation file has the same name but with .xml extension
        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            continue

        # Generate samples
        generate_positive_samples(image_path, annotation_path, pos_output_dir)
        generate_negative_samples(image_path, annotation_path, neg_output_dir, num_samples=15)

if __name__ == '__main__':
    # Define paths
    IMAGE_DIR = 'data/AgroPest-12/train_images'
    ANNOTATION_DIR = 'data/AgroPest-12/train_annotations'
    POS_OUTPUT_DIR = 'data/positives'
    NEG_OUTPUT_DIR = 'data/negatives'
    
    process_dataset(IMAGE_DIR, ANNOTATION_DIR, POS_OUTPUT_DIR, NEG_OUTPUT_DIR)
    print("Data preparation complete.")