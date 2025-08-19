import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler # New import
import numpy as np
import cv2
import os
from glob import glob
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, config, transform=None):
        self.image_paths = image_paths
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        
        if image is None:
            print(f"Warning: Could not read image at {path}. Returning placeholder.")
            placeholder_image = np.zeros((self.config["image_size"], self.config["image_size"], self.config["num_channels"]), dtype=np.uint8)
            image_tensor = self.transform(Image.fromarray(placeholder_image))
            return image_tensor, -1

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if self.transform:
            image = self.transform(image)
        
        label = os.path.basename(os.path.dirname(path))
        class_idx = self.config["class_names"].index(label)
        class_idx = torch.tensor(class_idx, dtype=torch.long)
        
        return image, class_idx

def get_train_transforms(config):
    """
    Defines the data augmentation and normalization pipeline for the training set.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms

def get_eval_transforms(config):
    """
    Defines the normalization pipeline for validation and test sets.
    """
    eval_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return eval_transforms

def get_train_val_loaders(base_path, config):
    """
    Loads training and validation data from separate folders.
    The training loader is now balanced using a WeightedRandomSampler.
    """
    train_x = glob(os.path.join(base_path, "train", "*", "*.jpg"))
    valid_x = glob(os.path.join(base_path, "val", "*", "*.jpg"))

    if not train_x or not valid_x:
        print("Warning: Train or validation directories are empty. Please check your paths.")
    
    train_transforms = get_train_transforms(config)
    eval_transforms = get_eval_transforms(config)

    train_dataset = CustomDataset(train_x, config, transform=train_transforms)
    valid_dataset = CustomDataset(valid_x, config, transform=eval_transforms)
    
    train_labels = [config["class_names"].index(os.path.basename(os.path.dirname(path))) for path in train_x]
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, 
        batch_size=config["batch_size"], 
        sampler=sampler, 
        pin_memory=True, 
        num_workers=4
    )
    
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, valid_loader

def get_benchmark_paths(base_path):
    """
    Loads all image paths for each benchmark and returns them in a dictionary.
    """
    benchmark_paths = {}
    test_benchmarks_path = os.path.join(base_path, "test")
    
    benchmark_folders = [d for d in os.listdir(test_benchmarks_path)
                         if os.path.isdir(os.path.join(test_benchmarks_path, d))]
    
    if not benchmark_folders:
        print("Warning: No benchmark folders found in the test directory.")
        return benchmark_paths

    for benchmark_name in benchmark_folders:
        benchmark_path = os.path.join(test_benchmarks_path, benchmark_name, "**", "*.jpg")
        all_images = glob(benchmark_path, recursive=True)
        benchmark_paths[benchmark_name] = all_images

    return benchmark_paths
