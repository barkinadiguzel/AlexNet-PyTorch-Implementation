import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. Data Transforms (Augmentation)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),  # central 256x256 patch
    transforms.RandomCrop(224),  # random 224x224 patch
    transforms.RandomHorizontalFlip(),  # horizontal reflections
    transforms.ToTensor(),
    transforms.Lambda(lambda x: pca_lighting(x)),  # PCA lighting
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


# 2. DataLoaders
def get_imagenet_loaders(train_dir, val_dir, batch_size=128, num_workers=4):
    train_dataset = ImageNetDataset(train_dir, transform=train_transforms)
    val_dataset = ImageNetDataset(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
