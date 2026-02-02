"""
Utilities for downloading and loading the EuroSAT dataset
"""

import os
import zipfile
import requests
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import FloodNetDataset


def download_dataset(data_dir="FloodNet"):
    """
    Download the FloodNet dataset if not already present.

    Args:
        data_dir: Directory where the dataset will be stored
    """
    if os.path.exists(data_dir):
        print(f"Dataset already available at: {data_dir}")
        return

    # FloodNet dataset URL
    url = "https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=4&dl=1"
    zip_path = "FloodNet.zip"

    print("Downloading FloodNet dataset ...")
    print(f"URL: {url}")

    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        downloaded = 0
        for data in response.iter_content(chunk_size=8192):
            f.write(data)
            downloaded += len(data)
            # Simple progress bar
            done = int(50 * downloaded / total_size)
            print(
                f"\r[{'=' * done}{' ' * (50 - done)}] " f"{downloaded/1e6:.1f}/{total_size/1e6:.1f} MB",
                end="",
                flush=True,
            )

    print("\n\nExtracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip file
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully!")


def load_images_path(data_dir: str = "FloodNet", max_per_class: int = 300) -> tuple[list[Path], list[Path], list[str]]:
    class_names = [
        "background",
        "building-flooded",
        "building-non-flooded",
        "road-flooded",
        "road-non-flooded",
        "water",
        "tree",
        "vehicle",
        "pool",
        "grass",
    ]

    data_path = Path(data_dir)
    data_path = data_path / "FloodNet-Supervised_v1.0"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}\n" f"Please run download_dataset() first.")

    image_files = []
    label_files = []
    for category in ["test", "train", "val"]:
        org_images_path = Path(data_path / category / f"{category}-org-img")
        label_images_path = Path(data_path / category / f"{category}-label-img")

        assert org_images_path.exists(), f"Image path not found: {org_images_path}"
        assert label_images_path.exists(), f"Label path not found: {label_images_path}"

        image_files_list = sorted(org_images_path.rglob("*.jpg"))
        label_files_list = sorted(label_images_path.rglob("*.png"))

        image_files += image_files_list  # Equivalente a extend()
        label_files += label_files_list

    return image_files, label_files, class_names


def create_label_mapping(class_names):
    """
    Create a mapping from class names to numeric indices.

    Args:
        class_names: List of class names

    Returns:
        dict: Mapping from class name to index
    """
    label_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    return label_to_idx


def labels_to_numeric(labels, label_to_idx):
    """
    Convert string labels to numeric indices.

    Args:
        labels: Array of string labels
        label_to_idx: Dictionary mapping labels to indices

    Returns:
        numpy.ndarray: Array of numeric labels
    """
    return np.array([label_to_idx[label] for label in labels])


def create_dataloaders(images_path, labels_path, class_names, config):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    This function:
    1. Converts string labels to numeric indices
    2. Creates the full dataset
    3. Splits into train/val/test sets
    4. Creates DataLoader objects for each split

    Args:
        images_path: List of image file paths
        labels_path: List of label file paths
        class_names: List of class names
        config: Configuration object with batch_size, train_ratio, etc.

    Returns:
        tuple: (train_loader, val_loader, test_loader, label_to_idx)
    """
    # Create label mapping
    label_to_idx = create_label_mapping(class_names)
    print(f"\nLabel mapping: {label_to_idx}")

    # Create dataset
    transform = FloodNetDataset.get_default_transform(image_size=config.IMAGE_SIZE)
    full_dataset = FloodNetDataset(images_path, labels_path, transform=transform)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = int(config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    print("DataLoaders created successfully!")

    return train_loader, val_loader, test_loader, label_to_idx
