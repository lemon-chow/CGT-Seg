import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import random
from pathlib import Path
import cv2  # 导入 OpenCV
import torch.nn.functional as F




def load_image(filename):
    return Image.open(filename)

class BasicDataset(Dataset):
    def __init__(self, images_dir: Path, mask_dir: Path, scale: float = 1.0, crop_size: int = None, use_erosion=False, erosion_size=3):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.crop_size = crop_size if crop_size is not None else 224  # 设置默认裁剪尺寸
        self.use_erosion = use_erosion
        self.erosion_size = erosion_size

        self.files = []
        for dirpath, _, filenames in os.walk(images_dir):
            for filename in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg")]:
                image_path = Path(dirpath) / filename
                relative_path = image_path.relative_to(images_dir)
                mask_file = relative_path.with_suffix('.png')
                mask_path = mask_dir / mask_file

                if mask_path.exists():
                    self.files.append((image_path, mask_path))
                else:
                    print(f"Mask file {mask_path} not found for image {image_path}")

        if not self.files:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        self.mask_values = self.calculate_unique_mask_values()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10)
        ])

    def calculate_unique_mask_values(self):
        unique_values = set()
        for _, mask_path in self.files:
            mask = np.asarray(Image.open(mask_path))
            unique_values.update(np.unique(mask))
        return sorted(list(unique_values))

    def __len__(self):
        return len(self.files)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img = np.asarray(pil_img)

        if is_mask:
        #     if self.use_erosion:
        #         # 转换为灰度图，如果不是灰度图
        #         if img.ndim == 3:
        #             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #         # 定义腐蚀操作的核
        #         kernel = np.ones((self.erosion_size, self.erosion_size), np.uint8)
        #         # 应用腐蚀操作
        #         img = cv2.erode(img, kernel, iterations=1)
                
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))  # Convert to C, H, W

            if (img > 1).any():
                img = img / 255.0
            return img

    def random_crop(self, img, mask):
        assert img.size == mask.size, \
            f'Image and mask should be the same size, but are {img.size} and {mask.size}'

        w, h = img.size
        th, tw = self.crop_size, self.crop_size

        if w == tw and h == th:
            return img, mask

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return img, mask

    def __getitem__(self, idx):
        try:
            image_path, mask_path = self.files[idx]
            img = load_image(image_path)
            mask = load_image(mask_path)
            assert img.size == mask.size, \
                f'Image and mask should be the same size, but are {img.size} and {mask.size}'

            if self.crop_size:
                img, mask = self.random_crop(img, mask)
            
            # Apply additional data augmentations to both image and mask
            # seed = np.random.randint(2147483647)  # Make a seed with numpy generator 
            # random.seed(seed)  # Apply this seed to img transforms
            # img = self.transform(img)
            # random.seed(seed)  # Apply the same seed to mask transforms
            # mask = self.transform(mask)

            img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True) 
            
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous().clone(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous().clone()
            }

        except Exception as e:
            print(f"Error processing file {self.files[idx]}: {e}")
            raise

class LungDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, crop_size=None, use_erosion=False, erosion_size=None):
        super().__init__(images_dir, mask_dir, scale, crop_size, use_erosion, erosion_size)
