# detection/nms.py
# This module implements the Non-Maximum Suppression algorithm to clean up
# overlapping bounding boxes.

import numpy as np

def non_maximum_suppression(boxes, overlap_thresh):
    """
    Performs Non-Maximum Suppression to filter out redundant, overlapping boxes.
    This is a vectorized and efficient implementation.

    Args:
        boxes (np.ndarray): An array of detections, with each row being
                            (startX, startY, endX, endY, confidence_score).
        overlap_thresh (float): The Intersection over Union (IoU) threshold
                                to use for suppressing boxes.

    Returns:
        np.ndarray: A filtered array of the final, non-overlapping boxes.
    """
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # If the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing division
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # Compute the area of the bounding boxes
    area = (x2 - x1) * (y2 - y1)
    
    # Sort the bounding boxes by their confidence score (descending)
    idxs = np.argsort(scores)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the top-left
        # corner and the smallest (x, y) coordinates for the
        # bottom-right corner of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute the ratio of overlap (Intersection over Union)
        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        overlap = intersection / union

        # Delete all indexes from the index list that have an IoU greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick]