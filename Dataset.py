import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

class PotholeDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, split_files, transform=None):
        """
        Args:
            img_dir (str): Path to the directory with images.
            annotations_dir (str): Path to the directory with labeled proposals.
            split_files (list): List of filenames for this dataset split.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.split_files = split_files
        self.transform = transform
        self.data = self._load_annotations()

    def _load_annotations(self):
        data = []
        for file_name in tqdm(self.split_files):
            img_name = file_name.replace(".xml", ".jpg")
            img_path = os.path.join(self.img_dir, img_name)
            annotation_path = os.path.join(self.annotations_dir, file_name.replace(".xml", "_labeled_proposals.txt"))
            
            if not os.path.exists(annotation_path):
                continue  # Skip files without annotation
            
            with open(annotation_path, 'r') as f:
                for line in f:
                    xmin, ymin, xmax, ymax, label = line.strip().split(',')
                    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
                    target = 1 if label == "pothole" else 0
                    data.append((img_path, bbox, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, bbox, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Crop the region of interest based on bbox
        img = img.crop(bbox)
        
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)