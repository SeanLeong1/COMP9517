import cv2
import numpy as np
import os
import glob
import sys
import time

def extract_and_save_sift_features():
    """
    (Task 1.1)
    - Loads all images from the 'train' directory.
    - Extracts SIFT features from all of them.
    - Consolidates them into one large numpy array.
    - Saves the array to the 'models' folder.
    """
    
    print("--- Task 1.1: SIFT Feature Extraction Started ---")
    
    # 1. 设置路径
    # (假设此 .py 文件位于 'features' 文件夹中)
    
    # 获取项目根目录 (COMP9517)
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 定义数据和输出路径
    DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, '..', 'data', 'AgroPest-12') # 指向 E:/comp9517_project/data/AgroPest-12
    TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train', 'images')
    OUTPUT_FILE = os.path.join(PROJECT_ROOT_DIR, 'models', 'sift_all_train_descriptors.npy')
    
    # 2. 检查路径
    if not os.path.exists(TRAIN_IMAGE_DIR):
        print(f"!!! 错误：找不到训练图像路径 !!!")
        print(f"请检查路径: {TRAIN_IMAGE_DIR}")
        return

    # 3. 获取所有图像路径
    image_paths = glob.glob(os.path.join(TRAIN_IMAGE_DIR, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(TRAIN_IMAGE_DIR, '*.png')))
    num_images = len(image_paths)
    
    if num_images == 0:
        print(f"!!! 警告：在 {TRAIN_IMAGE_DIR} 中没有找到图像。")
        return

    print(f"Total training images found: {num_images}")

    # 4. 初始化 SIFT
    sift = cv2.SIFT_create()
    all_descriptors_list = []
    start_time = time.time()

    print("Extracting features (这可能需要几分钟)...")
    
    # 5. 循环提取
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray_img, None) # [cite: 6546-6547, 6554, 6882-6891]
        
        if des is not None:
            all_descriptors_list.append(des)
            
        if (i + 1) % 500 == 0 or (i + 1) == num_images:
            print(f"Processed {i + 1}/{num_images} images...")

    print("...Feature extraction complete.")
    print(f"Total images with features: {len(all_descriptors_list)}")

    # 6. 整合并保存
    print("Consolidating all descriptors...")
    all_descriptors = np.vstack(all_descriptors_list)
    
    # 确保 models 文件夹存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 7. 保存
    np.save(OUTPUT_FILE, all_descriptors)
    
    end_time = time.time()
    print(f"\n✅ Task 1.1 COMPLETED in {end_time - start_time:.2f} seconds.")
    print(f"Total features extracted: {all_descriptors.shape[0]}")
    print(f"Data saved to: {OUTPUT_FILE}")
    print("--------------------------------------------------")

# 这是一个标准的 Python 写法：
# 只有当你直接运行 "python sift_bow_extractor.py" 时，下面的代码才会被执行
# 如果其他文件 import 这个文件，下面的代码不会运行
if __name__ == '__main__':
    extract_and_save_sift_features()