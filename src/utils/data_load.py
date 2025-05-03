import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from src.utils.normalización import search_paths
from src.utils.data_setup import download_and_get_dataset


def load_data(output_path: str, batch_size: int, train: int = 0.7, transform=None, augmentation=True, all_data=False):

    dataset_path = download_and_get_dataset()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    targets = np.array(dataset.targets)

    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=23)

    for train_idx, val_idx in splits.split(np.zeros(len(targets)), targets):
        train_indices = Subset(dataset, train_idx)
        val_indices = Subset(dataset, val_idx)

    train_loader = DataLoader(train_indices, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_indices, batch_size=32, shuffle=True)

    dataset_info(train_loader, "entrenamiento")
    dataset_info(test_loader, "prueba")

    return train_loader, test_loader

def dataset_info(dataset, msg):
    contador = {0: 0, 1: 0, 2:0}
    for _, label in dataset:
        contador[label] += 1
    print(f"Conteo de etiquetas en {msg}: {contador}")