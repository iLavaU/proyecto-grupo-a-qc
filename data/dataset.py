"""
FloodNet Dataset Module
=======================
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import settings


class FloodNetDataset(Dataset):
    """
    PyTorch Dataset for FloodNet satellite imagery.

    This dataset loads satellite images directly without dimensionality reduction,
    allowing the CNN to extract features from the raw images.

    Args:
        images_paths (list[Path]): List of images paths
        labels_paths (list[Path]): List of labels paths
        image_transforms (callable, optional): Transform to apply to images
        label_transforms (callable, optional): Transform to apply to labels
    """

    def __init__(self, images_paths, labels_paths, image_transforms=None, label_transforms=None):
        """
        Initialize the dataset.

        Args:
            images_paths: List or array of image file paths
            labels_paths: List or array of corresponding label file paths
            image_transforms: Optional torchvision transforms for images
            label_transforms: Optional torchvision transforms for labels
        """
        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

        # Default transform if none provided
        if self.image_transforms is None:
            self.image_transforms = self.get_image_transform()

        if self.label_transforms is None:
            self.label_transforms = self.get_label_transform()

    def __len__(self):
        """Return the total number of samples"""
        return len(self.images_paths)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (image_tensor, label_tensor) where both are torch.Tensor
        """
        # Get image and label paths
        img_path = self.images_paths[idx]
        label_path = self.labels_paths[idx]

        # Load image and label
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")  # Assuming label is a grayscale mask

        # Apply transforms to image and label
        if self.image_transforms:
            img = self.image_transforms(img)
        if self.label_transforms:
            mask = self.label_transforms(mask)

        return img, mask

    @staticmethod
    def get_label_transform(image_size=(64, 64)):
        return transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST), # 
                transforms.PILToTensor(), # Keep label as integer tensor
                MaskToLong(),  # Remove channel dim and convert to long
            ]
        )

    @staticmethod
    def get_image_transform(image_size=(64, 64)):
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # ImageNet normalization (standard for pre-trained models)
                transforms.Normalize(mean=settings.MEAN, std=settings.STD),
            ]
        )
    
    

    def get_class_distribution(self):
        """
        Calculate the distribution of classes in the dataset.

        Returns:
            dict: Mapping from label to count
        """
        unique, counts = torch.unique(torch.tensor(self.labels_paths), return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self):
        """String representation of the dataset"""
        return (
            f"FloodNetDataset(samples={len(self)}, "
            f"image_shape={self.images_paths[0].shape}, "
            f"num_classes={len(set(self.labels_paths))})"
        )

class MaskToLong:
    """Custom transform to convert mask tensor to long and remove channel dimension."""
    def __call__(self, mask):
        return mask.squeeze(0).long()