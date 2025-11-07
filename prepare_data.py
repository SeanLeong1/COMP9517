# prepare_data.py
# This script processes the raw AgroPest-12 dataset to generate positive
# and negative image samples for training an object detector.

import os
import cv2
import numpy as np
from tqdm import tqdm

def parse_txt_annotations(annotation_path, img_width, img_height):
    """
    Parses a YOLO-style .txt annotation file to extract bounding box coordinates.

    Args:
        annotation_path (str): Path to the .txt annotation file.
        img_width (int): The width of the corresponding image.
        img_height (int): The height of the corresponding image.

    Returns:
        list: A list of bounding box dictionaries in pixel coordinates.
    """
    boxes = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # YOLO format: class_id x_center y_center width height (normalized)
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # De-normalize coordinates to pixel values
            box_width = width_norm * img_width
            box_height = height_norm * img_height
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height

            # Convert center coordinates to top-left (xmin, ymin)
            xmin = int(x_center - (box_width / 2))
            ymin = int(y_center - (box_height / 2))
            xmax = int(x_center + (box_width / 2))
            ymax = int(y_center + (box_height / 2))

            box = {
                'label': class_id, # Using the class ID as the label
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
            }
            boxes.append(box)
    return boxes

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def generate_positive_samples(image, gt_boxes, image_basename, output_dir, target_size=(64, 128)):
    """
    Crops positive samples from an image using its ground-truth boxes.
    """
    for i, box in enumerate(gt_boxes):
        cropped = image[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, target_size)
            filename = f"{image_basename}_pos_{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), resized)

def generate_negative_samples(image, gt_boxes, image_basename, output_dir, target_size=(64, 128), num_samples=15):
    """
    Generates negative samples by randomly sampling non-overlapping patches.
    """
    height, width, _ = image.shape
    samples_generated = 0
    
    for _ in range(num_samples * 20): # Attempt more times to ensure we get enough samples
        if samples_generated >= num_samples:
            break
        
        if width <= target_size[0] or height <= target_size[1]:
            continue

        x = np.random.randint(0, width - target_size[0])
        y = np.random.randint(0, height - target_size[1])
        random_window = [x, y, x + target_size[0], y + target_size[1]]
        
        is_negative = True
        for box in gt_boxes:
            gt_box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            if calculate_iou(random_window, gt_box) > 0.05: # Stricter overlap threshold
                is_negative = False
                break
        
        if is_negative:
            cropped = image[y:y + target_size[1], x:x + target_size[0]]
            resized = cv2.resize(cropped, target_size)
            filename = f"{image_basename}_neg_{samples_generated}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), resized)
            samples_generated += 1

def process_dataset(image_dir, annotation_dir, pos_output_dir, neg_output_dir):
    """
    Main function to process the entire dataset.
    """
    os.makedirs(pos_output_dir, exist_ok=True)
    os.makedirs(neg_output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files):
        image_basename = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        annotation_file = image_basename + '.txt'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            continue
            
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        height, width, _ = image.shape
        
        gt_boxes = parse_txt_annotations(annotation_path, width, height)
        
        generate_positive_samples(image, gt_boxes, image_basename, pos_output_dir)
        generate_negative_samples(image, gt_boxes, image_basename, neg_output_dir)

if __name__ == '__main__':
    BASE_DATA_PATH = 'data/AgroPest-12' 
    
    IMAGE_DIR = os.path.join(BASE_DATA_PATH, 'train', 'images')
    ANNOTATION_DIR = os.path.join(BASE_DATA_PATH, 'train', 'labels')
    POS_OUTPUT_DIR = 'data/positives'
    NEG_OUTPUT_DIR = 'data/negatives'
    
    print("Starting data preparation...")
    process_dataset(IMAGE_DIR, ANNOTATION_DIR, POS_OUTPUT_DIR, NEG_OUTPUT_DIR)
    print("Data preparation complete.")