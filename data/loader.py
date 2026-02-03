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
from config import settings


def download_dataset(data_dir: str = "FloodNet") -> None:
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


def load_images_path(
    data_dir: Path = Path("FloodNet"), max_per_class: int = 300
) -> tuple[list[Path], list[Path], list[str]]:
    data_path = data_dir / "FloodNet-Supervised_v1.0"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}\n" f"Please run download_dataset() first.")

    image_files = []
    label_files = []
    for category in ["test", "train", "val"]:
        org_images_path = Path(data_path / category / f"{category}-org-img")
        label_images_path = Path(data_path / category / f"{category}-label-img")

        if not org_images_path.exists() or not label_images_path.exists():
            print(f"Warning: Skipping missing category '{category}', folder not found.")
            continue

        image_files_list = sorted(org_images_path.rglob("*.jpg"))
        label_files_list = sorted(label_images_path.rglob("*.png"))

        if len(image_files_list) != len(label_files_list):
            raise ValueError(
                f"Number of images and labels do not match in category '{category}': "
                f"{len(image_files_list)} images vs {len(label_files_list)} labels."
            )
        if len(image_files_list) == 0:
            print(f"Warning: No images found in category '{category}'.")
            continue

        # Limitar a max_per_class imágenes por clase
        if max_per_class > 0:
            image_files_list = image_files_list[:max_per_class]
            label_files_list = label_files_list[:max_per_class]

        image_files += image_files_list
        label_files += label_files_list

    return image_files, label_files

def label_to_idx() -> dict[str, int]:
    """
    Create a mapping from class names to indices using settings.CLASS_MAPPING.
    
    Returns:
        dict: Dictionary mapping class names to their corresponding indices
              Example: {'Background': 0, 'Building-Flooded': 1, ...}
    """
    return {name: idx for idx, name in settings.CLASS_MAPPING.items()}


def idx_to_label() -> dict[int, str]:
    """
    Create a mapping from indices to class names using settings.CLASS_MAPPING.
    
    Returns:
        dict: Dictionary mapping indices to their corresponding class names
              Example: {0: 'Background', 1: 'Building-Flooded', ...}
    """
    return settings.CLASS_MAPPING.copy()

def create_dataloaders(images_path: list[Path], labels_path: list[Path], config) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
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
        config: Configuration object with batch_size, train_ratio, etc.

    Returns:
        tuple: (train_loader, val_loader, test_loader, label_to_idx)
    """
    # Create label mapping
    label_to_idx_map = label_to_idx()
    print(f"\nLabel mapping: {label_to_idx_map}")

    # Create dataset
    images_transform = FloodNetDataset.get_image_transform(image_size=config.IMAGE_SIZE)
    labels_transform = FloodNetDataset.get_label_transform(image_size=config.IMAGE_SIZE)
    full_dataset = FloodNetDataset(images_path, labels_path, images_transform, labels_transform)

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

    return train_loader, val_loader, test_loader, label_to_idx_map
