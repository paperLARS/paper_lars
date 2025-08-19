import os
import torch
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'NIL': 0, 'POD': 1}
        self.idx_to_class = {0: 'NIL', 1: 'POD'}
        self._load_data()

    def _load_data(self):
        for class_name, class_index in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_index)

        print(f"Loaded {len(self.image_paths)} images from {self.root_dir}")
        label_counts = Counter(self.labels)
        class_counts = {self.idx_to_class[label]: count for label, count in label_counts.items()}
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  Class '{class_name}': {count} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def train_dataloader(data_dir, batch_size, img_size):
    """
    Train Dataloader with undersampling for imbalanced classes.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(root_dir=data_dir, transform=train_transform)

    # Calculate class counts
    class_counts = Counter(train_dataset.labels)
    num_samples = len(train_dataset)

    # Determine the minority class
    min_class_count = min(class_counts.values())
    print(f"Number of samples in minority class: {min_class_count}")
    
    class_weights = {cls: min_class_count / count for cls, count in class_counts.items()}
    weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    print(f"Total samples loaded: {len(train_loader)}")
    return train_loader

def val_dataloader(data_dir, batch_size, img_size):
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CustomDataset(root_dir=data_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader

def test_dataloader(data_dir, batch_size, img_size):
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = CustomDataset(root_dir=data_dir, transform=test_transform)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #return test_loader
    return test_dataset

            