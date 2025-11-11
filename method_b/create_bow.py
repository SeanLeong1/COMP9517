import argparse
import json
import os
import random
import sys
import warnings
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from joblib import load
from sklearn.exceptions import InconsistentVersionWarning

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import AgroPestDataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VOCABULARY_PATH = os.path.join(PROJECT_ROOT, "data", "method_b_models", "bow_vocabulary.pkl")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "method_b_models", "bow_datasets")
DEFAULT_NUM_INSECT_CLASSES = 12  # AgroPest-12 => classes indexed 0..11
DEFAULT_BACKGROUND_CLASS_ID = DEFAULT_NUM_INSECT_CLASSES
RANDOM_SEED = 1337

print(f"Loading vocabulary from {VOCABULARY_PATH} ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    kmeans = load(VOCABULARY_PATH)
cluster_centers = kmeans.cluster_centers_.astype(np.float32)
center_norms = np.sum(cluster_centers * cluster_centers, axis=1)
K = cluster_centers.shape[0]
print(f"KMeans vocabulary loaded with {K} clusters.")
sift = cv2.SIFT_create()


def bow_hist(descriptors: np.ndarray) -> np.ndarray:
    """Encode SIFT descriptors into a normalized histogram."""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(K, np.float32)
    descriptors = descriptors.astype(np.float32, copy=False)
    desc_norms = np.sum(descriptors * descriptors, axis=1, keepdims=True)
    dots = np.einsum("ij,kj->ik", descriptors, cluster_centers, optimize=True)
    distances = desc_norms + center_norms[None, :] - 2.0 * dots
    assignments = np.argmin(distances, axis=1)
    hist, _ = np.histogram(assignments, bins=np.arange(K + 1))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def intersection_over_union(a: Sequence[int], b: Sequence[int]) -> float:
    """IoU for inclusive pixel coordinates."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1) + 1)
    inter_h = max(0, min(ay2, by2) - max(ay1, by1) + 1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def random_bg_boxes(
    height: int,
    width: int,
    num_boxes: int,
    min_size: int,
    max_size: int,
    forbidden: Sequence[Sequence[int]],
    rng: random.Random,
) -> List[Tuple[int, int, int, int]]:
    """Sample low-overlap background boxes."""
    if height < 2 or width < 2 or num_boxes <= 0:
        return []

    cap = max(1, min(height, width))
    min_size = max(8, min(min_size, cap))
    max_size = max(min_size, min(max_size, cap))

    boxes: List[Tuple[int, int, int, int]] = []
    attempts = 0
    max_attempts = max(50, num_boxes * 25)

    forbid = [tuple(int(v) for v in box) for box in forbidden]

    while len(boxes) < num_boxes and attempts < max_attempts:
        attempts += 1
        size = rng.randint(min_size, max_size)
        max_x = max(0, width - size)
        max_y = max(0, height - size)
        x1 = rng.randint(0, max_x) if max_x > 0 else 0
        y1 = rng.randint(0, max_y) if max_y > 0 else 0
        x2 = min(width - 1, x1 + size - 1)
        y2 = min(height - 1, y1 + size - 1)
        candidate = (x1, y1, x2, y2)

        if (x2 - x1) < 7 or (y2 - y1) < 7:
            continue

        if all(intersection_over_union(candidate, fb) < 0.2 for fb in forbid) and all(
            intersection_over_union(candidate, prev) < 0.2 for prev in boxes
        ):
            boxes.append(candidate)

    return boxes


def crop_patch(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    """Crop image patch using inclusive bbox coordinates."""
    h, w = image.shape[:2]
    x1 = max(0, min(int(bbox[0]), w - 1))
    y1 = max(0, min(int(bbox[1]), h - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), w - 1))
    y2 = max(y1 + 1, min(int(bbox[3]), h - 1))
    return image[y1 : y2 + 1, x1 : x2 + 1]


def encode_patch(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    """Crop, extract SIFT descriptors, and encode BoW histogram."""
    patch = crop_patch(image, bbox)
    if patch.size == 0:
        return np.zeros(K, np.float32)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    _, descriptors = sift.detectAndCompute(gray, None)
    return bow_hist(descriptors)


def encode_split(
    split: str,
    *,
    bg_per_image: int,
    bg_min_size: int,
    bg_max_size: int,
    background_class_id: int,
    max_images: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict, set]:
    """Create BoW features for a dataset split."""
    loader = AgroPestDataLoader(root_dir=PROJECT_ROOT, subset_name=split)

    histograms: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict] = []
    stats = {"images_used": 0, "positives": 0, "background": 0}
    observed_classes = set()

    for idx in range(len(loader)):
        if max_images is not None and idx >= max_images:
            break
        item = loader[idx]
        image = item["image"]
        boxes = item["boxes"]
        cls_ids = item["labels"]
        img_path = item["path"]
        height, width = image.shape[:2]

        for bbox, cls_id in zip(boxes, cls_ids):
            bbox_tuple = tuple(int(v) for v in bbox)
            histograms.append(encode_patch(image, bbox_tuple))
            label_id = int(cls_id)
            labels.append(label_id)
            observed_classes.add(label_id)
            metadata.append(
                {
                    "split": split,
                    "image_path": img_path,
                    "bbox": [int(v) for v in bbox_tuple],
                    "class_id": label_id,
                    "class_name": f"Class {label_id}",
                    "sample_type": "positive",
                }
            )
            stats["positives"] += 1

        forbidden = boxes.tolist() if len(boxes) > 0 else []
        bg_boxes = random_bg_boxes(
            height,
            width,
            num_boxes=bg_per_image,
            min_size=bg_min_size,
            max_size=bg_max_size,
            forbidden=forbidden,
            rng=rng,
        )

        for bbox in bg_boxes:
            histograms.append(encode_patch(image, bbox))
            labels.append(background_class_id)
            metadata.append(
                {
                    "split": split,
                    "image_path": img_path,
                    "bbox": [int(v) for v in bbox],
                    "class_id": background_class_id,
                    "class_name": "Background",
                    "sample_type": "background",
                }
            )
            stats["background"] += 1

        stats["images_used"] += 1
        if (idx + 1) % 50 == 0:
            print(f"[{split}] Processed {idx + 1}/{len(loader)} images ...")

    if histograms:
        hist_array = np.stack(histograms).astype(np.float32)
    else:
        hist_array = np.zeros((0, K), dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)

    stats["samples"] = len(histograms)

    return hist_array, label_array, metadata, stats, observed_classes


def save_split(split: str, histograms: np.ndarray, labels: np.ndarray, metadata: List[Dict], output_dir: str) -> Dict[str, str]:
    """Persist histograms/labels arrays and metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, f"bow_{split}.npz")
    np.savez_compressed(feature_path, histograms=histograms, labels=labels)

    metadata_path = os.path.join(output_dir, f"bow_{split}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {"features": feature_path, "metadata": metadata_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BoW training data with positive and background samples.")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Dataset splits to encode (default: train).")
    parser.add_argument("--bg-per-image", type=int, default=6, help="Number of background crops per image.")
    parser.add_argument("--bg-min-size", type=int, default=80, help="Minimum background crop size.")
    parser.add_argument("--bg-max-size", type=int, default=192, help="Maximum background crop size.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to store BoW feature files.")
    parser.add_argument("--background-class-id", type=int, default=None, help="Label id to use for background samples.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images per split (debugging).")
    parser.add_argument(
        "--num-insect-classes",
        type=int,
        default=DEFAULT_NUM_INSECT_CLASSES,
        help="Total insect classes (used for label map completeness).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = os.path.abspath(args.output_dir)

    background_class_id = args.background_class_id if args.background_class_id is not None else DEFAULT_BACKGROUND_CLASS_ID
    known_class_ids = set(range(max(0, args.num_insect_classes)))

    if background_class_id in known_class_ids:
        raise ValueError("Background class id conflicts with an insect class id.")

    print(f"Background class id: {background_class_id}")
    print(f"Writing outputs to: {output_dir}")

    all_stats = {}
    observed_classes = set()

    for split in args.splits:
        (
            histograms,
            labels,
            metadata,
            stats,
            split_classes,
        ) = encode_split(
            split,
            bg_per_image=args.bg_per_image,
            bg_min_size=args.bg_min_size,
            bg_max_size=args.bg_max_size,
            background_class_id=background_class_id,
            max_images=args.max_images,
            rng=rng,
        )
        paths = save_split(split, histograms, labels, metadata, output_dir)
        stats.update(paths)
        all_stats[split] = stats
        observed_classes.update(split_classes)

        print(
            f"[{split}] samples={stats['samples']} (positives={stats['positives']}, "
            f"background={stats['background']}), files saved to {paths}"
        )

    # Persist label map for downstream usage
    label_map = {str(cls_id): f"Class {cls_id}" for cls_id in sorted(known_class_ids.union(observed_classes))}
    label_map[str(background_class_id)] = "Background"
    label_map_path = os.path.join(PROJECT_ROOT, "data", "method_b_models", "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {label_map_path}")

    stats_path = os.path.join(output_dir, "bow_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Split statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
