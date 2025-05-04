import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.data_setup import download_and_get_dataset


def load_data(output_path: str, batch_size: int, test: int = 0.3, transform=None, augmentation=True, all_data=False):

    print("[INFO] Cargando datos")
    dataset_path = download_and_get_dataset()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    targets = np.array(dataset.targets)

    splits = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=23)

    for train_idx, val_idx in splits.split(np.zeros(len(targets)), targets):
        train_indices = Subset(dataset, train_idx)
        val_indices = Subset(dataset, val_idx)

    train_loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_indices, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":
    load_data("output/", 32)