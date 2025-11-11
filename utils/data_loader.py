# utils/data_loader.py

import numpy as np
import cv2
from pathlib import Path

class AgroPestDataLoader:
    # 假设 root_dir 是 COMP9517 根目录的路径
    def __init__(self, root_dir, subset_name):
        # 1. 设置基础路径：指向 COMP9517/data/AgroPest-12
        project_root = Path(root_dir)
        base_dir = project_root / "data" / "AgroPest-12" 
        
        # 2. 定义图像和标签目录 (e.g., train/images 和 train/labels)
        self.image_dir = base_dir / subset_name / "images" 
        self.label_dir = base_dir / subset_name / "labels" 
        
        # 3. 获取所有图像文件列表
        # 假设图像文件是 .jpg 格式
        self.image_files = sorted(list(self.image_dir.glob("*.jpg"))) 
        
        if not self.image_files:
            raise FileNotFoundError(f"在 {self.image_dir} 中找不到任何图像文件。请检查数据是否已下载并正确放置。")
            
        print(f"成功初始化 {subset_name} 集，共 {len(self.image_files)} 个样本。")
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self._load_item(index)

    # ----------------------------------------------------
    # 核心方法将定义在下面...
    # ----------------------------------------------------
    def _load_yolo_label(self, image_path: Path, img_width: int, img_height: int):
        # 1. 构造标签文件的路径 (将图像文件的后缀名 .jpg 替换为 .txt)
        label_path = self.label_dir / (image_path.stem + ".txt")
        
        boxes = []
        labels = []
        
        if not label_path.exists():
            # 没有标签文件，说明图像中没有目标，返回空列表
            return np.array([]), np.array([])
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # 归一化坐标：x_center, y_center, width, height (0 到 1 之间)
                    xc_n, yc_n, w_n, h_n = [float(p) for p in parts[1:]]
                    
                    # 2. 核心步骤：反归一化 (转换为像素坐标)
                    
                    # 计算实际像素中心点
                    x_center = xc_n * img_width
                    y_center = yc_n * img_height
                    
                    # 计算实际像素宽度和高度
                    w = w_n * img_width
                    h = h_n * img_height
                    
                    # 转换为 (x_min, y_min, x_max, y_max) 格式 (左上角, 右下角)
                    x_min = int(x_center - w / 2)
                    y_min = int(y_center - h / 2)
                    x_max = int(x_center + w / 2)
                    y_max = int(y_center + h / 2)
                    
                    # 3. 边界约束：确保边界框不超出图像范围
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width - 1, x_max)
                    y_max = min(img_height - 1, y_max)
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

        return np.array(boxes, dtype=np.int32), np.array(labels, dtype=np.int32)
    def _load_item(self, index):
        # 1. 获取图像路径
        image_path = self.image_files[index]
        
        # 2. 读取图像 (使用 OpenCV，格式为 BGR，适合后续处理)
        image = cv2.imread(str(image_path)) 
        
        if image is None:
            raise IOError(f"无法读取图像：{image_path}")
        
        # 3. 获取图像尺寸 (OpenCV 图像是 H, W, C 格式)
        img_height, img_width, _ = image.shape
        
        # 4. 加载和解析标签
        boxes, labels = self._load_yolo_label(image_path, img_width, img_height)
        
        # 5. 返回结果
        return {
            "image": image,
            "boxes": boxes, 
            "labels": labels, 
            "path": str(image_path)
        }